from __future__ import annotations

import torch


def load_jit_model(jit_filepath: str, device: str = "cuda") -> torch.jit.ScriptModule:
    model = torch.jit.load(jit_filepath, map_location=device)
    return model.eval().to(device)


class ImageTokenizer(torch.nn.Module):
    def __init__(
        self,
        checkpoint_enc: str,
        checkpoint_dec: str,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self._enc_model = load_jit_model(checkpoint_enc, device)
        self._dec_model = load_jit_model(checkpoint_dec, device)

    def decode(self, input_latent: torch.Tensor) -> torch.Tensor:
        original_dtype = input_latent.dtype
        model_dtype = next(self.parameters()).dtype
        output_tensor = self._dec_model(input_latent.to(model_dtype))
        return output_tensor.to(original_dtype)

    def encode(self, input_tensor: torch.Tensor) -> torch.Tensor:
        original_dtype = input_tensor.dtype
        model_dtype = next(self.parameters()).dtype
        output_latent = self._enc_model(input_tensor.to(model_dtype))
        if isinstance(output_latent, torch.Tensor):
            return output_latent.to(original_dtype)
        return output_latent[0].to(original_dtype)
