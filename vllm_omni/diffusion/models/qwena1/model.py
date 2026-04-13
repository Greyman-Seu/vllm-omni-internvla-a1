from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch._dynamo as dynamo
import torch.nn.functional as F
from einops import rearrange
from huggingface_hub import snapshot_download
from safetensors.torch import load_model as load_model_as_safetensor
from torch import Tensor, nn
from transformers.models.auto import CONFIG_MAPPING

from .modeling_qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3VLTextModel, apply_rotary_pos_emb, eager_attention_forward

from .config import QwenA1Config
from .constants import DEFAULT_COSMOS_DIR, DEFAULT_COSMOS_REPO, OBS_IMAGES, OBS_PREFIX, OBS_STATE, OPENPI_ATTENTION_MASK_VALUE
from .cosmos import ImageTokenizer


def get_safe_dtype(target_dtype: torch.dtype, device_type: str) -> torch.dtype:
    if device_type == 'mps' and target_dtype == torch.float64:
        return torch.float32
    if device_type == 'cpu' and target_dtype == torch.bfloat16:
        return torch.float32
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device: torch.device,
) -> Tensor:
    if dimension % 2 != 0:
        raise ValueError(f'dimension ({dimension}) must be divisible by 2')
    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def pad_vector(vector: torch.Tensor, new_dim: int) -> torch.Tensor:
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


@dataclass
class SuffixStaticContext:
    state_emb: torch.Tensor
    full_att_2d_masks_4d: torch.Tensor
    position_ids: torch.Tensor


def compute_layer_complete(
    layer_idx: int,
    inputs_embeds: list[torch.Tensor],
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    und_expert: nn.Module,
    gen_expert: nn.Module,
    act_expert: nn.Module,
) -> list[torch.Tensor]:
    models = [und_expert.language_model, gen_expert, act_expert]
    query_states = []
    key_states = []
    value_states = []
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        hidden_states = layer.input_layernorm(hidden_states)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        if layer.self_attn.q_proj.weight.dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype=torch.bfloat16)
        query_state = layer.self_attn.q_norm(layer.self_attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_state = layer.self_attn.k_norm(layer.self_attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)

    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)
    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = und_expert.model.language_model.rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = apply_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        unsqueeze_dim=1,
    )
    scaling = und_expert.language_model.layers[layer_idx].self_attn.scaling
    att_output, _ = eager_attention_forward(
        und_expert.language_model.layers[layer_idx].self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )

    head_dim = und_expert.language_model.layers[layer_idx].self_attn.head_dim
    num_attention_heads = und_expert.language_model.layers[layer_idx].self_attn.config.num_attention_heads
    batch_size = query_states.shape[0]
    att_output = att_output.reshape(batch_size, -1, num_attention_heads * head_dim)

    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]
        att_chunk = att_output[:, start_pos:end_pos]
        if att_chunk.dtype != layer.self_attn.o_proj.weight.dtype:
            att_chunk = att_chunk.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_chunk)
        out_emb = out_emb + hidden_states
        residual = out_emb
        out_emb = layer.post_attention_layernorm(out_emb)
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        out_emb = out_emb + residual
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds


class QwenConfig:
    def __init__(
        self,
        head_dim: int,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_hidden_layers: int,
        num_key_value_heads: int,
    ) -> None:
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads


def get_qwen_config(variant: str) -> QwenConfig:
    num_hidden_layers = int(variant.split('_')[-1][:-1])
    if variant.startswith('qwen3_vl'):
        return QwenConfig(128, 2048, 6144, 16, num_hidden_layers, 8)
    if variant.startswith('qwen3'):
        return QwenConfig(128, 1024, 3072, 16, num_hidden_layers, 8)
    raise ValueError(f'Unknown variant: {variant}')


class Qwen3VLWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config: QwenConfig,
        action_expert_config: QwenConfig,
        precision: Literal['bfloat16', 'float32'] = 'bfloat16',
    ) -> None:
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING['qwen3_vl']()
        vlm_config_hf.text_config.hidden_size = vlm_config.hidden_size
        vlm_config_hf.text_config.intermediate_size = vlm_config.intermediate_size
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_attention_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.num_hidden_layers
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_key_value_heads
        vlm_config_hf.text_config.max_position_embeddings = 262144
        vlm_config_hf.text_config.rope_scaling = {
            'mrope_interleaved': True,
            'mrope_section': [24, 20, 20],
            'rope_type': 'default',
        }
        vlm_config_hf.text_config.tie_word_embeddings = True
        vlm_config_hf.tie_word_embeddings = True
        vlm_config_hf.vision_config.deepstack_visual_indexes = [5, 11, 17]
        vlm_config_hf.vision_config.depth = 24
        vlm_config_hf.vision_config.hidden_size = 1024
        vlm_config_hf.vision_config.intermediate_size = 4096
        vlm_config_hf.vision_config.out_hidden_size = 2048

        self.und_expert = Qwen3VLForConditionalGeneration(config=vlm_config_hf)

        gen_expert_config_hf = CONFIG_MAPPING['qwen3_vl_text']()
        gen_expert_config_hf.head_dim = action_expert_config.head_dim
        gen_expert_config_hf.hidden_size = action_expert_config.hidden_size
        gen_expert_config_hf.intermediate_size = action_expert_config.intermediate_size
        gen_expert_config_hf.num_attention_heads = action_expert_config.num_attention_heads
        gen_expert_config_hf.num_hidden_layers = action_expert_config.num_hidden_layers
        gen_expert_config_hf.num_key_value_heads = action_expert_config.num_key_value_heads
        gen_expert_config_hf.max_position_embeddings = self.und_expert.config.text_config.max_position_embeddings
        gen_expert_config_hf.rope_scaling = self.und_expert.config.text_config.rope_scaling
        self.gen_expert = Qwen3VLTextModel(config=gen_expert_config_hf)
        self.gen_expert.embed_tokens = None
        self.gen_expert.lm_head = None

        act_expert_config_hf = CONFIG_MAPPING['qwen3_vl_text']()
        act_expert_config_hf.head_dim = action_expert_config.head_dim
        act_expert_config_hf.hidden_size = action_expert_config.hidden_size
        act_expert_config_hf.intermediate_size = action_expert_config.intermediate_size
        act_expert_config_hf.num_attention_heads = action_expert_config.num_attention_heads
        act_expert_config_hf.num_hidden_layers = action_expert_config.num_hidden_layers
        act_expert_config_hf.num_key_value_heads = action_expert_config.num_key_value_heads
        act_expert_config_hf.max_position_embeddings = self.und_expert.config.text_config.max_position_embeddings
        act_expert_config_hf.rope_scaling = self.und_expert.config.text_config.rope_scaling
        self.act_expert = Qwen3VLTextModel(config=act_expert_config_hf)
        self.act_expert.embed_tokens = None
        self.act_expert.lm_head = None

        self.to_selected_precision(precision)

    def to_selected_precision(self, precision: Literal['bfloat16', 'float32']) -> None:
        if precision == 'bfloat16':
            self.to(dtype=torch.bfloat16)
        elif precision == 'float32':
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f'Unsupported precision: {precision}')

        keep_fp32 = ['input_layernorm', 'post_attention_layernorm', 'model.norm']
        for name, param in self.named_parameters():
            if any(key in name for key in keep_fp32):
                param.data = param.data.to(dtype=torch.float32)

    def forward(
        self,
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor | None,
        past_key_values: Any,
        inputs_embeds: list[torch.Tensor | None],
        use_cache: bool,
    ) -> tuple[list[torch.Tensor | None], Any]:
        if inputs_embeds[1] is None and inputs_embeds[2] is None:
            prefix_output = self.und_expert.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            return [prefix_output.last_hidden_state, None, None], prefix_output.past_key_values

        if inputs_embeds[0] is None and inputs_embeds[2] is None:
            middle_output = self.gen_expert.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            return [None, middle_output.last_hidden_state, None], middle_output.past_key_values

        if inputs_embeds[0] is None and inputs_embeds[1] is None:
            suffix_output = self.act_expert.forward(
                inputs_embeds=inputs_embeds[2],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            return [None, None, suffix_output.last_hidden_state], None

        models = [self.und_expert.language_model, self.gen_expert, self.act_expert]
        stacked_inputs = [inputs_embeds[0], inputs_embeds[1], inputs_embeds[2]]
        for layer_idx in range(self.und_expert.config.text_config.num_hidden_layers):
            stacked_inputs = compute_layer_complete(
                layer_idx,
                stacked_inputs,
                attention_mask,
                position_ids,
                und_expert=self.und_expert,
                gen_expert=self.gen_expert,
                act_expert=self.act_expert,
            )
        outputs = [models[i].norm(stacked_inputs[i]) for i in range(3)]
        return outputs, None


class QwenA1(nn.Module):
    def __init__(self, config: QwenA1Config) -> None:
        super().__init__()
        self.config = config

        vlm_config = get_qwen_config(config.qwen3_vl_variant)
        action_expert_config = get_qwen_config(config.action_expert_variant)
        self.qwen3_vl_with_expert = Qwen3VLWithExpertModel(vlm_config, action_expert_config, precision=config.dtype)

        if not (DEFAULT_COSMOS_DIR / 'encoder.jit').exists():
            snapshot_download(repo_id=DEFAULT_COSMOS_REPO, local_dir=str(DEFAULT_COSMOS_DIR))
        self.cosmos = ImageTokenizer(
            checkpoint_enc=str(DEFAULT_COSMOS_DIR / 'encoder.jit'),
            checkpoint_dec=str(DEFAULT_COSMOS_DIR / 'decoder.jit'),
            device=config.device,
        )

        hidden_size = action_expert_config.hidden_size
        vae_dim = 16
        ds = config.scale_factor
        self.cosmos_in_proj = nn.Conv2d(vae_dim, hidden_size, kernel_size=1, stride=1, padding=0)
        self.downsample_conv = nn.Conv2d(hidden_size, hidden_size, kernel_size=ds, stride=ds, padding=0)
        self.upsample_conv = nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=ds, stride=ds, padding=0)
        self.cosmos_out_proj = nn.Linear(hidden_size, vae_dim)
        self.cosmos_out_layer_norm = nn.LayerNorm(hidden_size)

        self.action_in_proj = nn.Linear(config.max_action_dim, hidden_size)
        self.action_out_proj = nn.Linear(hidden_size, config.max_action_dim)
        self.state_proj = nn.Linear(config.max_state_dim, hidden_size)
        self.action_time_mlp_in = nn.Linear(2 * hidden_size, hidden_size)
        self.action_time_mlp_out = nn.Linear(hidden_size, hidden_size)

        if config.compile_model:
            torch.set_float32_matmul_precision('high')
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)

        self.cosmos.eval()
        for param in self.cosmos.parameters():
            param.requires_grad = False

        self.set_attention_implementation(config.attn_implementation)

    def set_attention_implementation(self, attn_implementation: str) -> None:
        self.config.attn_implementation = attn_implementation
        self.qwen3_vl_with_expert.und_expert.config.text_config._attn_implementation = attn_implementation
        self.qwen3_vl_with_expert.und_expert.language_model.config._attn_implementation = attn_implementation
        self.qwen3_vl_with_expert.gen_expert.config._attn_implementation = attn_implementation
        self.qwen3_vl_with_expert.act_expert.config._attn_implementation = attn_implementation

    def _prepare_attention_masks_4d(
        self,
        att_2d_masks: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        bias_dtype = torch.float32 if self.config.attn_implementation == "eager" else dtype
        valid_bias = torch.zeros((), dtype=bias_dtype, device=att_2d_masks.device)
        invalid_bias = torch.tensor(
            OPENPI_ATTENTION_MASK_VALUE,
            dtype=bias_dtype,
            device=att_2d_masks.device,
        )
        return torch.where(att_2d_masks_4d, valid_bias, invalid_bias)

    def sample_noise(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        return torch.normal(0.0, 1.0, shape, dtype=torch.float32, device=device)

    @dynamo.disable
    def embed_prefix(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_token_id = self.qwen3_vl_with_expert.und_expert.config.image_token_id
        pixel_values = pixel_values.view(-1, pixel_values.shape[-1])
        image_grid_thw = image_grid_thw.view(-1, 3)
        image_embs, _ = self.qwen3_vl_with_expert.und_expert.visual(pixel_values, image_grid_thw)

        embs = self.qwen3_vl_with_expert.und_expert.get_input_embeddings()(lang_tokens)
        batch_size, seq_len, hidden_dim = embs.shape
        embs = embs.view(-1, hidden_dim)
        lang_tokens_flat = lang_tokens.view(-1)
        embs[lang_tokens_flat == image_token_id] = image_embs
        embs = embs.view(batch_size, seq_len, hidden_dim)

        pad_masks = lang_masks.to(torch.bool)
        att_masks = torch.zeros_like(pad_masks, dtype=torch.bool, device=pad_masks.device)
        return embs, pad_masks, att_masks

    def get_cosmos_features(self, images: torch.Tensor) -> torch.Tensor:
        shape = images.shape[:-3]
        channels, height, width = images.shape[-3:]
        images = images.reshape(-1, channels, height, width)
        if (height, width) != (256, 256):
            images = F.interpolate(images, size=(256, 256), mode='bilinear', align_corners=False)
        images = images * 2 - 1
        features = self.cosmos.encode(images)
        channels, height, width = features.shape[-3:]
        return features.view(*shape, channels, height, width)

    def embed_middle(self, images: torch.Tensor, img_masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = images.device
        batch_size, n_view, timesteps = images.shape[:3]
        features = self.get_cosmos_features(images)
        features = rearrange(features, 'b n t c h w -> (b n t) c h w')
        features = self.cosmos_in_proj(features)
        features = self.downsample_conv(features)
        features = rearrange(features, '(b n t) c h w -> b n t c h w', b=batch_size, n=n_view, t=timesteps)
        self.cosmos_feat_shape = features.shape

        _, _, _, _, height, width = features.shape
        embs = rearrange(features, 'b n t c h w -> b (n t h w) c')
        pad_masks = torch.zeros((batch_size, n_view, timesteps, height, width), dtype=torch.bool, device=device)
        pad_masks[img_masks] = True
        pad_masks = rearrange(pad_masks, 'b n t h w -> b (n t h w)')
        att_masks = torch.tensor([1] + [0] * (embs.shape[1] - 1), dtype=torch.bool, device=device)
        att_masks = att_masks[None, :].expand(batch_size, att_masks.shape[0])
        return embs, pad_masks, att_masks

    def prepare_suffix_static_context(
        self,
        state: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        max_prefix_position_ids: torch.Tensor,
    ) -> SuffixStaticContext:
        if self.state_proj.weight.dtype == torch.float32:
            state = state.to(torch.float32)

        state_emb = self.state_proj(state)
        device = state_emb.device
        batch_size = state_emb.shape[0]
        suffix_len = self.config.chunk_size + 1
        prefix_len = prefix_pad_masks.shape[1]

        suffix_pad_masks = torch.ones(batch_size, suffix_len, dtype=torch.bool, device=device)
        suffix_att_masks = torch.tensor([1] + [1] + [0] * (self.config.chunk_size - 1), dtype=torch.bool, device=device)
        suffix_att_masks = suffix_att_masks[None, :].expand(batch_size, suffix_len)
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(
            torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2),
            dtype=state_emb.dtype,
        )
        position_ids = (
            torch.arange(1, suffix_len + 1, device=max_prefix_position_ids.device)
            .repeat(3, 1, 1)
            .to(max_prefix_position_ids)
            + max_prefix_position_ids
        )
        return SuffixStaticContext(
            state_emb=state_emb,
            full_att_2d_masks_4d=full_att_2d_masks_4d,
            position_ids=position_ids,
        )

    def embed_suffix(
        self,
        state: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.state_proj.weight.dtype == torch.float32:
            state = state.to(torch.float32)

        state_emb = self.state_proj(state)
        batch_size = state_emb.shape[0]
        device = state_emb.device
        state_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)

        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        ).to(dtype=timestep.dtype)
        action_emb = self.action_in_proj(noisy_actions)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        action_time_emb = self.action_time_mlp_out(F.silu(self.action_time_mlp_in(action_time_emb)))
        action_mask = torch.ones(batch_size, action_time_emb.shape[1], dtype=torch.bool, device=device)

        embs = torch.cat([state_emb[:, None, :], action_time_emb], dim=1)
        pad_masks = torch.cat([state_mask, action_mask], dim=1)
        att_masks = torch.tensor([1] + [1] + [0] * (self.config.chunk_size - 1), dtype=embs.dtype, device=device)
        att_masks = att_masks[None, :].expand(batch_size, self.config.chunk_size + 1)
        return embs, pad_masks, att_masks

    def get_position_ids(
        self,
        lang_tokens: torch.Tensor,
        image_grid_thw: torch.Tensor | None,
        pad_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, Any]:
        seq_len = lang_tokens.shape[1]
        padded_lang_tokens = torch.ones_like(pad_masks).to(lang_tokens) * 777
        padded_lang_tokens[:, :seq_len] = lang_tokens
        attention_mask = pad_masks.to(lang_tokens)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.view(-1, 3)
        return self.qwen3_vl_with_expert.und_expert.model.get_rope_index(
            padded_lang_tokens,
            image_grid_thw,
            attention_mask=attention_mask,
        )

    @torch.no_grad()
    def sample_actions(
        self,
        images: torch.Tensor,
        img_masks: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        num_steps: int | None = None,
        decode_image: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if num_steps is None:
            num_steps = self.config.num_inference_steps
        batch_size = state.shape[0]
        device = state.device
        dtype = state.dtype
        if noise is None:
            noise = self.sample_noise((batch_size, self.config.chunk_size, self.config.max_action_dim), device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(pixel_values, image_grid_thw, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids, _ = self.get_position_ids(lang_tokens, image_grid_thw, prefix_pad_masks)
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(
            prefix_att_2d_masks,
            dtype=prefix_embs.dtype,
        )
        _, past_key_values = self.qwen3_vl_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None, None],
            use_cache=True,
        )
        max_prefix_position_ids = prefix_position_ids.max(dim=-1, keepdim=True).values

        middle_embs, middle_pad_masks, middle_att_masks = self.embed_middle(images[:, :, :2], img_masks)
        middle_len = middle_pad_masks.shape[1]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, middle_len, prefix_len)
        middle_att_2d_masks = make_att_2d_masks(middle_pad_masks, middle_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, middle_att_2d_masks], dim=2)
        middle_position_ids = (
            torch.arange(1, middle_len + 1, device=max_prefix_position_ids.device)
            .repeat(3, 1, 1)
            .to(max_prefix_position_ids)
            + max_prefix_position_ids
        )
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(
            full_att_2d_masks,
            dtype=middle_embs.dtype,
        )
        (_, middle_out, _), past_key_values = self.qwen3_vl_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=middle_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, middle_embs, None],
            use_cache=True,
        )

        max_position_ids = middle_position_ids.max(dim=-1, keepdim=True).values
        curr_pad_masks = torch.cat([prefix_pad_masks, middle_pad_masks], dim=1)
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        suffix_static = None
        if self.config.enable_suffix_static_context_optimization:
            suffix_static = self.prepare_suffix_static_context(state, curr_pad_masks, max_position_ids)
        while time >= -dt / 2:
            expanded_time = time.expand(batch_size)
            if suffix_static is None:
                v_t = self.denoise_step(state, curr_pad_masks, past_key_values, max_position_ids, x_t.to(dtype), expanded_time.to(dtype))
            else:
                v_t = self.denoise_step_optimized(suffix_static, past_key_values, x_t.to(dtype), expanded_time.to(dtype))
            x_t = x_t + dt * v_t
            time += dt

        return x_t, None if not decode_image else middle_out

    def denoise_step(
        self,
        state: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        past_key_values: Any,
        max_prefix_position_ids: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, timestep)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        position_ids = (
            torch.arange(1, suffix_len + 1, device=max_prefix_position_ids.device)
            .repeat(3, 1, 1)
            .to(max_prefix_position_ids)
            + max_prefix_position_ids
        )
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(
            full_att_2d_masks,
            dtype=suffix_embs.dtype,
        )
        outputs_embeds, _ = self.qwen3_vl_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, None, suffix_embs],
            use_cache=False,
        )
        suffix_out = outputs_embeds[2][:, -self.config.chunk_size:].to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

    def denoise_step_optimized(
        self,
        suffix_static: SuffixStaticContext,
        past_key_values: Any,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        state_emb = suffix_static.state_emb
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        ).to(dtype=timestep.dtype)
        action_emb = self.action_in_proj(x_t)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        action_time_emb = self.action_time_mlp_out(F.silu(self.action_time_mlp_in(action_time_emb)))
        suffix_embs = torch.cat([state_emb[:, None, :], action_time_emb], dim=1)
        outputs_embeds, _ = self.qwen3_vl_with_expert.forward(
            attention_mask=suffix_static.full_att_2d_masks_4d,
            position_ids=suffix_static.position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, None, suffix_embs],
            use_cache=False,
        )
        suffix_out = outputs_embeds[2][:, -self.config.chunk_size:].to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


class StandaloneQwenA1Policy(nn.Module):
    def __init__(self, config: QwenA1Config) -> None:
        super().__init__()
        self.config = config
        self.model = QwenA1(config)
        self.model.to(config.device)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str | Path,
        *,
        config: QwenA1Config | None = None,
        strict: bool = False,
    ) -> 'StandaloneQwenA1Policy':
        checkpoint_dir = Path(checkpoint_dir)
        if config is None:
            config = QwenA1Config.from_pretrained(checkpoint_dir)
        instance = cls(config)
        load_model_as_safetensor(instance, str(checkpoint_dir / 'model.safetensors'), strict=strict, device=config.device)
        instance.eval()
        return instance

    def to(self, *args, **kwargs):  # type: ignore[override]
        super().to(*args, **kwargs)
        self.model.cosmos.to(torch.bfloat16)
        self.model.action_out_proj.to(torch.float32)
        return self

    def _preprocess_images(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        images = torch.stack([batch[f'{OBS_IMAGES}.image{i}'] for i in range(3)], dim=1)
        img_masks = torch.stack([batch[f'{OBS_IMAGES}.image{i}_mask'] for i in range(3)], dim=1)
        return images, img_masks

    def prepare_state(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return pad_vector(batch[OBS_STATE], self.config.max_state_dim)

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, torch.Tensor],
        *,
        decode_image: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        pixel_values = batch[f'{OBS_PREFIX}pixel_values']
        image_grid_thw = batch[f'{OBS_PREFIX}image_grid_thw']
        lang_tokens = batch[f'{OBS_PREFIX}input_ids']
        lang_masks = batch[f'{OBS_PREFIX}attention_mask']
        state = self.prepare_state(batch)
        images, img_masks = self._preprocess_images(batch)
        return self.model.sample_actions(
            images,
            img_masks,
            pixel_values,
            image_grid_thw,
            lang_tokens,
            lang_masks,
            state,
            decode_image=decode_image,
        )
