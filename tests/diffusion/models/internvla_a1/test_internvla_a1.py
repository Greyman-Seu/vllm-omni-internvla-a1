# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for lightweight InternVLA-A1 helpers and pipeline branches."""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from transformers.models.auto import CONFIG_MAPPING

import vllm_omni.diffusion.models.internvla_a1.pipeline_internvla_a1 as internvla_pipeline_module
from vllm_omni.diffusion.models.internvla_a1.adapter_qwen3_vl import (
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextRMSNorm,
)
from vllm_omni.diffusion.models.internvla_a1.config import (
    DEFAULT_COSMOS_REPO,
    OBS_IMAGES,
    OBS_STATE,
    OBS_TASK,
    OPENPI_ATTENTION_MASK_VALUE,
    InternVLAA1Config,
    InternVLAA1TrainMetadata,
)
from vllm_omni.diffusion.models.internvla_a1.cosmos_ci_torch import build_cosmos_ci_torch_model
from vllm_omni.diffusion.models.internvla_a1.model_cosmos import (
    ImageTokenizer,
    infer_cosmos_ci_spatial_compression,
    is_safetensors_checkpoint,
    load_cosmos_component,
)
from vllm_omni.diffusion.models.internvla_a1.model_internvla_a1 import (
    InternVLAA1,
    InternVLAA1Policy,
    create_sinusoidal_pos_embedding,
    get_qwen_config,
    get_safe_dtype,
    make_att_2d_masks,
    pad_vector,
    resize_with_pad,
    resolve_cosmos_checkpoint_paths,
)
from vllm_omni.diffusion.models.internvla_a1.pipeline_internvla_a1 import (
    InternVLAA1Pipeline,
    get_internvla_a1_post_process_func,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestInternVLAA1Config:
    def test_from_model_config_filters_unknown_fields_and_normalizes_resolution(self):
        config = InternVLAA1Config.from_model_config(
            {
                "dtype": "float32",
                "image_resolution": [128, 256],
                "max_state_dim": 12,
                "unknown_key": "ignored",
            }
        )

        assert config.dtype == "float32"
        assert config.image_resolution == (128, 256)
        assert config.max_state_dim == 12
        assert not hasattr(config, "unknown_key")

    def test_from_model_config_uses_defaults_for_empty_input(self):
        config = InternVLAA1Config.from_model_config(None)
        assert config == InternVLAA1Config()

    def test_from_pretrained_loads_config_json(self, tmp_path):
        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "dtype": "float32",
                    "image_resolution": [64, 96],
                    "chunk_size": 8,
                }
            ),
            encoding="utf-8",
        )

        config = InternVLAA1Config.from_pretrained(tmp_path)

        assert config.dtype == "float32"
        assert config.image_resolution == (64, 96)
        assert config.chunk_size == 8


class TestInternVLAA1TrainMetadata:
    def test_from_pretrained_uses_defaults_when_train_config_missing(self, tmp_path):
        metadata = InternVLAA1TrainMetadata.from_pretrained(tmp_path)

        assert metadata == InternVLAA1TrainMetadata()

    def test_from_pretrained_extracts_processor_model_name(self, tmp_path):
        (tmp_path / "train_config.json").write_text(
            json.dumps(
                {
                    "dataset": {
                        "action_mode": "absolute",
                        "data_transforms": {
                            "inputs": [
                                {"type": "noop"},
                                {
                                    "type": "internvla_a1_processor",
                                    "pretrained_model_name_or_path": "local/qwen3-vl",
                                },
                            ]
                        },
                    }
                }
            ),
            encoding="utf-8",
        )

        metadata = InternVLAA1TrainMetadata.from_pretrained(tmp_path)

        assert metadata.action_mode == "absolute"
        assert metadata.processor_model_name == "local/qwen3-vl"


class TestInternVLAA1Helpers:
    @pytest.mark.parametrize(
        ("target_dtype", "device_type", "expected"),
        [
            (torch.float64, "mps", torch.float32),
            (torch.bfloat16, "cpu", torch.float32),
            (torch.float32, "cpu", torch.float32),
        ],
    )
    def test_get_safe_dtype(self, target_dtype, device_type, expected):
        assert get_safe_dtype(target_dtype, device_type) == expected

    def test_create_sinusoidal_pos_embedding_zero_time_has_expected_pattern(self):
        time = torch.zeros(2, dtype=torch.float32)

        embedding = create_sinusoidal_pos_embedding(
            time,
            dimension=6,
            min_period=1e-3,
            max_period=1.0,
            device=torch.device("cpu"),
        )

        assert embedding.shape == (2, 6)
        assert torch.allclose(embedding[:, :3], torch.zeros(2, 3), atol=1e-6)
        assert torch.allclose(embedding[:, 3:], torch.ones(2, 3), atol=1e-6)

    def test_create_sinusoidal_pos_embedding_rejects_odd_dimension(self):
        with pytest.raises(ValueError, match="must be divisible by 2"):
            create_sinusoidal_pos_embedding(
                torch.tensor([0.0]),
                dimension=5,
                min_period=1e-3,
                max_period=1.0,
                device=torch.device("cpu"),
            )

    def test_make_att_2d_masks_combines_schedule_and_padding(self):
        pad_masks = torch.tensor([[True, True, True, False]])
        att_masks = torch.tensor([[True, True, False, False]])

        result = make_att_2d_masks(pad_masks, att_masks)

        expected = torch.tensor(
            [
                [
                    [True, False, False, False],
                    [True, True, True, False],
                    [True, True, True, False],
                    [False, False, False, False],
                ]
            ]
        )
        assert torch.equal(result, expected)

    def test_pad_vector_zero_pads_to_requested_size(self):
        vector = torch.tensor([[1.0, 2.0]])

        padded = pad_vector(vector, 5)

        assert padded.shape == (1, 5)
        assert torch.equal(padded[:, :2], vector)
        assert torch.equal(padded[:, 2:], torch.zeros(1, 3))

    def test_pad_vector_returns_original_when_dim_is_already_large_enough(self):
        vector = torch.randn(2, 4)

        padded = pad_vector(vector, 3)

        assert padded is vector

    def test_resize_with_pad_preserves_aspect_ratio_and_zero_pads(self):
        images = torch.ones(2, 3, 2, 4)

        resized = resize_with_pad(images, (8, 8))

        assert resized.shape == (2, 3, 8, 8)
        assert torch.allclose(resized[:, :, 2:6, :], torch.ones(2, 3, 4, 8), atol=1e-6)
        assert torch.equal(resized[:, :, :2, :], torch.zeros(2, 3, 2, 8))
        assert torch.equal(resized[:, :, 6:, :], torch.zeros(2, 3, 2, 8))

    def test_resize_with_pad_rejects_non_4d_input(self):
        with pytest.raises(ValueError, match="Expected \\[T, C, H, W\\]"):
            resize_with_pad(torch.ones(3, 8, 8), (16, 16))

    def test_get_qwen_config_supports_known_variants(self):
        qwen3_vl = get_qwen_config("qwen3_vl_28l")
        qwen3 = get_qwen_config("qwen3_28l")

        assert qwen3_vl.hidden_size == 2048
        assert qwen3_vl.num_hidden_layers == 28
        assert qwen3.hidden_size == 1024
        assert qwen3.num_hidden_layers == 28

    def test_get_qwen_config_rejects_unknown_variant(self):
        with pytest.raises(ValueError, match="Unknown variant"):
            get_qwen_config("internvl_unknown")

    def test_resolve_cosmos_checkpoint_paths_returns_existing_local_files(self, monkeypatch, tmp_path):
        encoder = tmp_path / "encoder.safetensors"
        decoder = tmp_path / "decoder.safetensors"
        encoder.write_bytes(b"encoder")
        decoder.write_bytes(b"decoder")
        monkeypatch.setattr(
            "vllm_omni.diffusion.models.internvla_a1.model_internvla_a1.DEFAULT_COSMOS_DIR",
            tmp_path,
        )

        resolved_encoder, resolved_decoder = resolve_cosmos_checkpoint_paths()

        assert resolved_encoder == encoder
        assert resolved_decoder == decoder

    def test_resolve_cosmos_checkpoint_paths_raises_clear_error_when_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "vllm_omni.diffusion.models.internvla_a1.model_internvla_a1.DEFAULT_COSMOS_DIR",
            tmp_path,
        )

        with pytest.raises(FileNotFoundError, match="INTERNVLA_A1_COSMOS_DIR"):
            resolve_cosmos_checkpoint_paths()

        with pytest.raises(FileNotFoundError, match=DEFAULT_COSMOS_REPO):
            resolve_cosmos_checkpoint_paths()

    def test_resolve_cosmos_checkpoint_paths_honors_explicit_env_overrides(self, monkeypatch, tmp_path):
        encoder = tmp_path / "encoder.safetensors"
        decoder = tmp_path / "decoder.safetensors"
        encoder.write_bytes(b"encoder")
        decoder.write_bytes(b"decoder")
        monkeypatch.setenv("INTERNVLA_A1_COSMOS_ENCODER_PATH", str(encoder))
        monkeypatch.setenv("INTERNVLA_A1_COSMOS_DECODER_PATH", str(decoder))
        monkeypatch.setattr(
            "vllm_omni.diffusion.models.internvla_a1.model_internvla_a1.DEFAULT_COSMOS_DIR",
            tmp_path / "missing",
        )

        resolved_encoder, resolved_decoder = resolve_cosmos_checkpoint_paths()

        assert resolved_encoder == encoder
        assert resolved_decoder == decoder

    def test_resolve_cosmos_checkpoint_paths_uses_safetensors_defaults(self, monkeypatch, tmp_path):
        encoder = tmp_path / "encoder.safetensors"
        decoder = tmp_path / "decoder.safetensors"
        encoder.write_bytes(b"encoder")
        decoder.write_bytes(b"decoder")
        monkeypatch.setattr(
            "vllm_omni.diffusion.models.internvla_a1.model_internvla_a1.DEFAULT_COSMOS_DIR",
            tmp_path,
        )

        resolved_encoder, resolved_decoder = resolve_cosmos_checkpoint_paths()

        assert resolved_encoder == encoder
        assert resolved_decoder == decoder


class TestInternVLAA1CosmosHelpers:
    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("encoder.safetensors", True),
            ("decoder.safetensors", True),
            ("encoder.pt", False),
        ],
    )
    def test_is_safetensors_checkpoint(self, path, expected):
        assert is_safetensors_checkpoint(path) is expected

    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("Cosmos-Tokenizer-CI8x8/encoder.safetensors", 8),
            ("Cosmos-0.1-Tokenizer-CI16x16/decoder.safetensors", 16),
        ],
    )
    def test_infer_cosmos_ci_spatial_compression(self, path, expected):
        assert infer_cosmos_ci_spatial_compression(path) == expected

    def test_infer_cosmos_ci_spatial_compression_rejects_unknown_path(self):
        with pytest.raises(ValueError, match="Unable to infer"):
            infer_cosmos_ci_spatial_compression("cosmos_encoder.pt")

    def test_build_cosmos_ci_torch_model_exposes_encoder_and_decoder(self):
        model = build_cosmos_ci_torch_model(spatial_compression=8)

        encoder = model.encoder_module()
        decoder = model.decoder_module()

        assert isinstance(encoder, torch.nn.Sequential)
        assert isinstance(decoder, torch.nn.Sequential)

    def test_load_cosmos_component_supports_safetensors_torch_backend(self, tmp_path):
        checkpoint = tmp_path / "Cosmos-Tokenizer-CI8x8-encoder.safetensors"
        save_file({"dummy": torch.zeros(1)}, checkpoint)

        model = load_cosmos_component(
            checkpoint,
            component="encoder",
            device="cpu",
        )

        assert isinstance(model, torch.nn.Module)

    def test_image_tokenizer_torch_backend_accepts_safetensors_paths(self, tmp_path):
        encoder = tmp_path / "Cosmos-Tokenizer-CI8x8-encoder.safetensors"
        decoder = tmp_path / "Cosmos-Tokenizer-CI8x8-decoder.safetensors"
        save_file({"dummy": torch.zeros(1)}, encoder)
        save_file({"dummy": torch.zeros(1)}, decoder)

        tokenizer = ImageTokenizer(
            str(encoder),
            str(decoder),
            device="cpu",
        )

        assert isinstance(tokenizer._enc_model, torch.nn.Module)
        assert isinstance(tokenizer._dec_model, torch.nn.Module)

    def test_load_cosmos_component_rejects_non_safetensors(self, tmp_path):
        checkpoint = tmp_path / "Cosmos-Tokenizer-CI8x8-encoder.ckpt"
        checkpoint.write_bytes(b"ckpt")

        with pytest.raises(ValueError, match="expects `.safetensors`"):
            load_cosmos_component(
                checkpoint,
                component="encoder",
                device="cpu",
            )


class TestInternVLAA1AdapterOverrides:
    def test_text_decoder_layer_uses_local_rmsnorm_overrides(self):
        config = CONFIG_MAPPING["qwen3_vl_text"]()
        layer = Qwen3VLTextDecoderLayer(config, layer_idx=0)

        assert isinstance(layer.self_attn.q_norm, Qwen3VLTextRMSNorm)
        assert isinstance(layer.self_attn.k_norm, Qwen3VLTextRMSNorm)
        assert isinstance(layer.input_layernorm, Qwen3VLTextRMSNorm)
        assert isinstance(layer.post_attention_layernorm, Qwen3VLTextRMSNorm)


class TestInternVLAA1ModelHelpers:
    def test_prepare_attention_masks_4d_uses_float32_bias_for_eager(self):
        model = object.__new__(InternVLAA1)
        model.config = SimpleNamespace(attn_implementation="eager")
        att_2d_masks = torch.tensor([[[True, False], [True, True]]])

        result = model._prepare_attention_masks_4d(att_2d_masks, dtype=torch.bfloat16)

        assert result.shape == (1, 1, 2, 2)
        assert result.dtype == torch.float32
        assert result[0, 0, 0, 0].item() == 0.0
        assert result[0, 0, 0, 1].item() == pytest.approx(OPENPI_ATTENTION_MASK_VALUE)

    def test_prepare_attention_masks_4d_uses_requested_dtype_for_non_eager(self):
        model = object.__new__(InternVLAA1)
        model.config = SimpleNamespace(attn_implementation="sdpa")
        att_2d_masks = torch.tensor([[[True, False], [False, True]]])

        result = model._prepare_attention_masks_4d(att_2d_masks, dtype=torch.bfloat16)

        assert result.dtype == torch.bfloat16
        assert result[0, 0, 0, 0].item() == 0.0
        assert result[0, 0, 0, 1].item() == pytest.approx(OPENPI_ATTENTION_MASK_VALUE, rel=1e-2)

    def test_prepare_suffix_static_context_builds_expected_shapes(self):
        model = object.__new__(InternVLAA1)
        model.config = SimpleNamespace(chunk_size=3, attn_implementation="eager")
        model.state_proj = torch.nn.Linear(3, 4, bias=False)

        context = model.prepare_suffix_static_context(
            state=torch.tensor([[1.0, 2.0, 3.0]]),
            prefix_pad_masks=torch.tensor([[True, True, False]]),
            max_prefix_position_ids=torch.tensor([[[5]], [[5]], [[5]]]),
        )

        assert context.state_emb.shape == (1, 4)
        assert context.full_att_2d_masks_4d.shape == (1, 1, 4, 7)
        assert context.position_ids.shape == (3, 1, 4)
        assert torch.equal(context.position_ids[:, 0, 0], torch.tensor([6, 6, 6]))
        assert torch.equal(context.position_ids[:, 0, -1], torch.tensor([9, 9, 9]))


class TestInternVLAA1PolicyHelpers:
    def test_get_task_accepts_string_and_singleton_list(self):
        policy = object.__new__(InternVLAA1Policy)

        assert policy._get_task({OBS_TASK: "pick up cube"}) == "pick up cube"
        assert policy._get_task({OBS_TASK: ["move arm"]}) == "move arm"

    def test_get_task_rejects_invalid_batch_shapes_and_types(self):
        policy = object.__new__(InternVLAA1Policy)

        with pytest.raises(ValueError, match="only supports bs=1"):
            policy._get_task({OBS_TASK: ["a", "b"]})

        with pytest.raises(TypeError, match="Expected task string"):
            policy._get_task({OBS_TASK: [123]})

        with pytest.raises(TypeError, match="Unsupported task payload type"):
            policy._get_task({OBS_TASK: {"task": "bad"}})

    def test_prepare_resized_histories_requires_batch_size_one(self):
        policy = object.__new__(InternVLAA1Policy)
        policy.config = SimpleNamespace(image_resolution=(32, 32))

        batch = {
            f"{OBS_IMAGES}.image0": torch.zeros(2, 2, 3, 8, 8),
            f"{OBS_IMAGES}.image1": torch.zeros(1, 2, 3, 8, 8),
            f"{OBS_IMAGES}.image2": torch.zeros(1, 2, 3, 8, 8),
        }

        with pytest.raises(ValueError, match="only supports bs=1"):
            policy._prepare_resized_histories(batch)

    def test_prepare_state_pads_to_max_state_dim(self):
        policy = object.__new__(InternVLAA1Policy)
        policy.config = SimpleNamespace(max_state_dim=5)

        state = policy.prepare_state({OBS_STATE: torch.tensor([[1.0, 2.0, 3.0]])})

        assert state.shape == (1, 5)
        assert torch.equal(state[:, :3], torch.tensor([[1.0, 2.0, 3.0]]))
        assert torch.equal(state[:, 3:], torch.zeros(1, 2))

    def test_preprocess_images_stacks_histories_and_masks(self):
        policy = object.__new__(InternVLAA1Policy)
        resized_histories = [
            torch.ones(2, 3, 8, 8),
            torch.full((2, 3, 8, 8), 2.0),
            torch.full((2, 3, 8, 8), 3.0),
        ]
        batch = {
            f"{OBS_IMAGES}.image0_mask": torch.tensor([True]),
            f"{OBS_IMAGES}.image1_mask": torch.tensor([False]),
            f"{OBS_IMAGES}.image2_mask": torch.tensor([True]),
        }

        images, img_masks = policy._preprocess_images(batch, resized_histories=resized_histories)

        assert images.shape == (1, 3, 2, 3, 8, 8)
        assert torch.equal(images[0, 0], resized_histories[0])
        assert torch.equal(images[0, 1], resized_histories[1])
        assert torch.equal(images[0, 2], resized_histories[2])
        assert torch.equal(img_masks, torch.tensor([[True, False, True]]))

    def test_prepare_qwen_prefix_inputs_moves_tensors_to_model_device(self):
        policy = object.__new__(InternVLAA1Policy)
        policy.input_builder = SimpleNamespace(
            build=lambda histories, task: {
                "observation.pixel_values": torch.ones(2, 5, dtype=torch.float32),
                "observation.image_grid_thw": torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long),
                "observation.input_ids": torch.tensor([7, 8, 9], dtype=torch.long),
                "observation.attention_mask": torch.tensor([1, 1, 0], dtype=torch.long),
            }
        )
        policy.model = SimpleNamespace(
            action_in_proj=SimpleNamespace(weight=torch.zeros(1, dtype=torch.float16, device="cpu"))
        )

        pixel_values, image_grid_thw, input_ids, attention_mask = policy._prepare_qwen_prefix_inputs(
            resized_histories=[torch.zeros(2, 3, 4, 4)] * 3,
            task="stack blocks",
        )

        assert pixel_values.shape == (1, 2, 5)
        assert pixel_values.dtype == torch.float16
        assert image_grid_thw.shape == (1, 2, 3)
        assert input_ids.shape == (1, 3)
        assert attention_mask.shape == (1, 3)


class TestInternVLAA1PipelineHelpers:
    def test_pipeline_init_preserves_prefix(self, monkeypatch):
        monkeypatch.setattr(InternVLAA1Pipeline, "_build_config", lambda self, od_config: SimpleNamespace())
        monkeypatch.setattr(InternVLAA1Pipeline, "setup_diffusion_pipeline_profiler", lambda self, **kwargs: None)
        monkeypatch.setattr(InternVLAA1Pipeline, "_initialize_policy", lambda self: SimpleNamespace())
        monkeypatch.setattr(InternVLAA1Pipeline, "_setup_policy_profiler_targets", lambda self: None)
        monkeypatch.setattr(InternVLAA1Pipeline, "_warmup", lambda self: None)
        od_config = SimpleNamespace(
            model="/tmp/internvla-a1",
            custom_pipeline_args={},
            enable_diffusion_pipeline_profiler=False,
        )

        pipeline = InternVLAA1Pipeline(od_config=od_config, prefix="internvla_a1")

        assert pipeline.prefix == "internvla_a1"

    def test_post_process_func_is_identity(self):
        post_process = get_internvla_a1_post_process_func(SimpleNamespace())
        value = torch.tensor([1.0, 2.0])
        assert torch.equal(post_process(value), value)

    def test_load_config_dict_prefers_model_config_payload(self):
        pipeline = object.__new__(InternVLAA1Pipeline)
        od_config = SimpleNamespace(model="/unused", model_config={"dtype": "float32", "chunk_size": 4})

        config_dict = pipeline._load_config_dict(od_config)

        assert config_dict == {"dtype": "float32", "chunk_size": 4}

    def test_load_config_dict_reads_config_file_when_present(self, tmp_path):
        pipeline = object.__new__(InternVLAA1Pipeline)
        (tmp_path / "config.json").write_text(json.dumps({"dtype": "float32"}), encoding="utf-8")
        od_config = SimpleNamespace(model=str(tmp_path), model_config={})

        config_dict = pipeline._load_config_dict(od_config)

        assert config_dict == {"dtype": "float32"}

    def test_build_config_applies_custom_pipeline_overrides(self):
        pipeline = object.__new__(InternVLAA1Pipeline)
        pipeline._load_config_dict = lambda od_config: {"dtype": "bfloat16", "device": "cuda", "chunk_size": 4}
        od_config = SimpleNamespace(
            model=None,
            model_config={},
            dtype=torch.float16,
            custom_pipeline_args={
                "device": "cpu",
                "dtype": "float32",
                "compile_model": True,
                "attn_implementation": "sdpa",
                "enable_regional_compile": True,
                "regional_compile_dynamic": False,
            },
        )

        config = pipeline._build_config(od_config)

        assert config.device == "cpu"
        assert config.dtype == "float32"
        assert config.chunk_size == 4
        assert config.compile_model is True
        assert config.attn_implementation == "sdpa"
        assert config.enable_regional_compile is True
        assert config.regional_compile_dynamic is False

    def test_build_fake_batch_inputs_matches_config_shapes(self):
        pipeline = object.__new__(InternVLAA1Pipeline)
        pipeline.config = SimpleNamespace(
            device="cpu",
            dtype="float32",
            image_resolution=(16, 24),
            max_state_dim=7,
        )

        batch_inputs = pipeline._build_fake_batch_inputs()

        assert batch_inputs[OBS_STATE].shape == (1, 7)
        assert batch_inputs[OBS_TASK] == [""]
        assert batch_inputs[f"{OBS_IMAGES}.image0"].shape == (1, 2, 3, 16, 24)
        assert batch_inputs[f"{OBS_IMAGES}.image1_mask"].dtype == torch.bool

    def test_runtime_mode_reports_checkpoint_presence(self, tmp_path):
        pipeline = object.__new__(InternVLAA1Pipeline)
        pipeline.model_dir = str(tmp_path)

        assert pipeline.runtime_mode() == "no_checkpoint_policy"

        (tmp_path / "model.safetensors").write_bytes(b"")
        assert pipeline.runtime_mode() == "real_checkpoint_loaded"

    def test_apply_policy_optimizations_sets_attention_impl_and_compiles_targets(self, monkeypatch):
        compile_calls = []

        def fake_regionally_compile(module, dynamic):
            compile_calls.append((module.name, dynamic))

        monkeypatch.setattr(internvla_pipeline_module, "regionally_compile", fake_regionally_compile)

        def set_attention_implementation(value):
            fake_model.attn_implementation = value

        fake_model = SimpleNamespace(
            set_attention_implementation=set_attention_implementation,
            qwen3_vl_with_expert=SimpleNamespace(
                und_expert=SimpleNamespace(
                    visual=SimpleNamespace(name="visual"),
                    language_model=SimpleNamespace(name="language_model"),
                ),
                gen_expert=SimpleNamespace(name="gen_expert"),
                act_expert=SimpleNamespace(name="act_expert"),
            ),
        )
        policy = SimpleNamespace(model=fake_model)
        pipeline = object.__new__(InternVLAA1Pipeline)
        pipeline.config = SimpleNamespace(
            attn_implementation="sdpa",
            enable_regional_compile=True,
            regional_compile_dynamic=False,
        )

        pipeline._apply_policy_optimizations(policy)

        assert fake_model.attn_implementation == "sdpa"
        assert compile_calls == [
            ("visual", False),
            ("language_model", False),
            ("gen_expert", False),
            ("act_expert", False),
        ]

    def test_warmup_builds_zero_noise_and_invokes_policy(self):
        recorded = {}

        class DummyPolicy:
            def forward(self, batch_inputs, noise=None, decode_image=False):
                recorded["batch_inputs"] = batch_inputs
                recorded["noise"] = noise
                recorded["decode_image"] = decode_image
                return torch.zeros(1, 2, 3), None

        pipeline = object.__new__(InternVLAA1Pipeline)
        pipeline.config = SimpleNamespace(
            device="cpu",
            dtype="float32",
            image_resolution=(8, 8),
            max_state_dim=4,
            chunk_size=2,
            max_action_dim=3,
        )
        pipeline.policy = DummyPolicy()

        pipeline._warmup()

        assert recorded["decode_image"] is False
        assert recorded["noise"].shape == (1, 2, 3)
        assert torch.count_nonzero(recorded["noise"]) == 0
        assert recorded["batch_inputs"][OBS_STATE].shape == (1, 4)

    def test_warmup_swallows_policy_failures(self):
        class FailingPolicy:
            def forward(self, batch_inputs, noise=None, decode_image=False):
                del batch_inputs, noise, decode_image
                raise RuntimeError("warmup failed")

        pipeline = object.__new__(InternVLAA1Pipeline)
        pipeline.config = SimpleNamespace(
            device="cpu",
            dtype="float32",
            image_resolution=(8, 8),
            max_state_dim=4,
            chunk_size=2,
            max_action_dim=3,
        )
        pipeline.policy = FailingPolicy()

        pipeline._warmup()

    def test_forward_returns_error_when_batch_inputs_missing(self):
        pipeline = object.__new__(InternVLAA1Pipeline)
        pipeline.od_config = SimpleNamespace()

        req = SimpleNamespace(
            prompts=["prompt"],
            sampling_params=SimpleNamespace(extra_args={}),
        )

        output = pipeline.forward(req)

        assert output.output is None
        assert "sampling_params.extra_args['batch_inputs']" in output.error

    def test_forward_returns_model_output_and_decoded_payload(self):
        pipeline = object.__new__(InternVLAA1Pipeline)
        pipeline.od_config = SimpleNamespace()
        expected_actions = torch.ones(1, 3, 2)
        expected_decoded = torch.zeros(1, 3, 4, 4)
        pipeline._predict_actions = lambda batch_inputs, noise=None, decode_image=False: (
            expected_actions,
            expected_decoded,
        )

        req = SimpleNamespace(
            prompts=["prompt"],
            sampling_params=SimpleNamespace(
                extra_args={
                    "batch_inputs": {"obs": "value"},
                    "noise": torch.randn(1, 3, 2),
                    "decode_image": True,
                }
            ),
        )

        output = pipeline.forward(req)

        assert torch.equal(output.output, expected_actions)
        assert torch.equal(output.custom_output["decoded"], expected_decoded)
