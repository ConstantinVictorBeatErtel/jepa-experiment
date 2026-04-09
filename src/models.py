"""Tiny vision models for JEPA-style latent prediction and masked reconstruction."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from src.patching import batch_index_select, full_patch_indices, patchify, random_masking
from src.utils import load_checkpoint


def gather_positions(position_embeddings: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather learned positional embeddings for a subset of patch indices."""
    expanded = position_embeddings.expand(indices.size(0), -1, -1)
    gather_index = indices.unsqueeze(-1).expand(-1, -1, expanded.size(-1))
    return torch.gather(expanded, dim=1, index=gather_index)


@dataclass
class EncoderSpec:
    """Reusable encoder construction parameters."""

    image_size: int
    patch_size: int
    in_channels: int
    embed_dim: int
    depth: int
    num_heads: int
    mlp_ratio: float
    dropout: float


class TinyViTEncoder(nn.Module):
    """A compact ViT-style encoder that can process full or partial patch sets."""

    def __init__(self, spec: EncoderSpec):
        super().__init__()
        self.spec = spec
        self.image_size = spec.image_size
        self.patch_size = spec.patch_size
        self.in_channels = spec.in_channels
        self.embed_dim = spec.embed_dim
        self.num_patches = (spec.image_size // spec.patch_size) ** 2
        self.patch_dim = spec.in_channels * spec.patch_size * spec.patch_size

        self.patch_embed = nn.Linear(self.patch_dim, self.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=spec.num_heads,
            dim_feedforward=int(self.embed_dim * spec.mlp_ratio),
            dropout=spec.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=spec.depth)
        self.norm = nn.LayerNorm(self.embed_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        nn.init.zeros_(self.patch_embed.bias)

    def forward_tokens(self, patches: torch.Tensor, patch_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode flattened patches and return token and pooled features."""
        x = self.patch_embed(patches) + gather_positions(self.pos_embed, patch_indices)
        x = self.transformer(x)
        x = self.norm(x)
        return {"tokens": x, "pooled": x.mean(dim=1)}

    def forward_full(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode a full image using all patches."""
        patches = patchify(images, self.patch_size)
        indices = full_patch_indices(images.size(0), self.num_patches, images.device)
        return self.forward_tokens(patches, indices)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Return a single embedding per image."""
        return self.forward_full(images)["pooled"]


class MaskConditionedPredictor(nn.Module):
    """Predict masked outputs from a pooled context representation plus mask positions."""

    def __init__(self, embed_dim: int, hidden_dim: int, num_patches: int, output_dim: int):
        super().__init__()
        self.mask_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        nn.init.trunc_normal_(self.mask_pos_embed, std=0.02)

    def forward(self, context_tokens: torch.Tensor, masked_indices: torch.Tensor) -> torch.Tensor:
        """Predict one masked output per masked patch index."""
        context_summary = context_tokens.mean(dim=1)
        context_summary = context_summary.unsqueeze(1).expand(-1, masked_indices.size(1), -1)
        mask_pos = gather_positions(self.mask_pos_embed, masked_indices)
        return self.net(torch.cat([context_summary, mask_pos], dim=-1))


class JEPAStyleModel(nn.Module):
    """Tiny JEPA-style model with stop-gradient target features."""

    def __init__(
        self,
        encoder: TinyViTEncoder,
        predictor_hidden_dim: int,
        target_encoder_mode: str = "shared",
        ema_momentum: float = 0.996,
    ):
        super().__init__()
        self.context_encoder = encoder
        self.target_encoder_mode = target_encoder_mode
        self.ema_momentum = ema_momentum
        self.predictor = MaskConditionedPredictor(
            embed_dim=encoder.embed_dim,
            hidden_dim=predictor_hidden_dim,
            num_patches=encoder.num_patches,
            output_dim=encoder.embed_dim,
        )
        if target_encoder_mode == "ema":
            self.target_encoder = copy.deepcopy(encoder)
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad = False
        elif target_encoder_mode != "shared":
            raise ValueError("target_encoder_mode must be 'shared' or 'ema'.")

    def _active_target_encoder(self) -> TinyViTEncoder:
        if self.target_encoder_mode == "shared":
            return self.context_encoder
        return self.target_encoder

    @torch.no_grad()
    def update_ema_target(self) -> None:
        """EMA update for the optional target encoder."""
        if self.target_encoder_mode != "ema":
            return
        for target_param, source_param in zip(
            self.target_encoder.parameters(), self.context_encoder.parameters()
        ):
            target_param.data.mul_(self.ema_momentum).add_(source_param.data, alpha=1.0 - self.ema_momentum)

    def forward(self, images: torch.Tensor, mask_ratio: float) -> Dict[str, torch.Tensor]:
        """Compute masked latent predictions and stop-gradient targets."""
        patches = patchify(images, self.context_encoder.patch_size)
        visible_idx, masked_idx = random_masking(
            batch_size=images.size(0),
            num_patches=self.context_encoder.num_patches,
            mask_ratio=mask_ratio,
            device=images.device,
        )
        visible_patches = batch_index_select(patches, visible_idx)
        masked_patches = batch_index_select(patches, masked_idx)

        context_outputs = self.context_encoder.forward_tokens(visible_patches, visible_idx)
        predicted_latents = self.predictor(context_outputs["tokens"], masked_idx)

        with torch.no_grad():
            target_outputs = self._active_target_encoder().forward_tokens(masked_patches, masked_idx)

        return {
            "predictions": predicted_latents,
            "targets": target_outputs["tokens"].detach(),
            "visible_idx": visible_idx,
            "masked_idx": masked_idx,
        }

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Expose the encoder embedding for downstream evaluation."""
        return self.context_encoder.encode(images)


class MaskedPatchReconstructionModel(nn.Module):
    """Tiny masked-patch reconstruction baseline with a shared encoder."""

    def __init__(self, encoder: TinyViTEncoder, decoder_hidden_dim: int):
        super().__init__()
        self.context_encoder = encoder
        self.decoder = MaskConditionedPredictor(
            embed_dim=encoder.embed_dim,
            hidden_dim=decoder_hidden_dim,
            num_patches=encoder.num_patches,
            output_dim=encoder.patch_dim,
        )

    def forward(self, images: torch.Tensor, mask_ratio: float) -> Dict[str, torch.Tensor]:
        """Predict masked patch pixels from visible context."""
        patches = patchify(images, self.context_encoder.patch_size)
        visible_idx, masked_idx = random_masking(
            batch_size=images.size(0),
            num_patches=self.context_encoder.num_patches,
            mask_ratio=mask_ratio,
            device=images.device,
        )
        visible_patches = batch_index_select(patches, visible_idx)
        masked_patches = batch_index_select(patches, masked_idx)

        context_outputs = self.context_encoder.forward_tokens(visible_patches, visible_idx)
        reconstructed_patches = self.decoder(context_outputs["tokens"], masked_idx)

        return {
            "predictions": reconstructed_patches,
            "targets": masked_patches,
            "visible_idx": visible_idx,
            "masked_idx": masked_idx,
        }

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Expose the encoder embedding for downstream evaluation."""
        return self.context_encoder.encode(images)


def build_encoder(config: Dict) -> TinyViTEncoder:
    """Construct an encoder from the shared config."""
    spec = EncoderSpec(
        image_size=int(config["dataset"]["image_size"]),
        patch_size=int(config["model"]["patch_size"]),
        in_channels=int(config["dataset"].get("channels", 3)),
        embed_dim=int(config["model"]["embed_dim"]),
        depth=int(config["model"]["depth"]),
        num_heads=int(config["model"]["num_heads"]),
        mlp_ratio=float(config["model"]["mlp_ratio"]),
        dropout=float(config["model"]["dropout"]),
    )
    return TinyViTEncoder(spec)


def build_jepa_model(config: Dict) -> JEPAStyleModel:
    """Construct the JEPA-style pretraining model."""
    encoder = build_encoder(config)
    return JEPAStyleModel(
        encoder=encoder,
        predictor_hidden_dim=int(config["model"]["predictor_hidden_dim"]),
        target_encoder_mode=config["model"].get("target_encoder_mode", "shared"),
        ema_momentum=float(config["model"].get("ema_momentum", 0.996)),
    )


def build_mae_model(config: Dict) -> MaskedPatchReconstructionModel:
    """Construct the masked reconstruction baseline."""
    encoder = build_encoder(config)
    return MaskedPatchReconstructionModel(
        encoder=encoder,
        decoder_hidden_dim=int(config["model"]["decoder_hidden_dim"]),
    )


def get_encoder(
    checkpoint_path: str,
    device: Optional[str] = None,
    eval_mode: bool = True,
) -> TinyViTEncoder:
    """Load a reusable encoder from a JEPA or MAE checkpoint."""
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    encoder_spec = EncoderSpec(**checkpoint["encoder_config"])
    encoder = TinyViTEncoder(encoder_spec)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    if device is not None:
        encoder = encoder.to(device)
    if eval_mode:
        encoder.eval()
    return encoder
