# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BioIR adapter for fixed-coordinate Boltz-2 affinity scoring."""

from pathlib import Path

import torch
from torch import nn

from bionemo_ir._torch.modules.boltz.affinity import compute_distogram, create_cross_pair_mask
from bionemo_ir.models.boltz2 import Boltz2Affinity


def fixed_coordinates(coordinates: torch.Tensor) -> torch.Tensor:
    if coordinates.ndim == 4:
        if coordinates.shape[1] != 1:
            raise ValueError("Fixed-pose scoring requires one pose per record.")
        coordinates = coordinates[:, 0]
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError("Expected coordinates with shape (batch, atoms, 3).")
    return coordinates


def relative_position_features(model, features):
    relative = model.get_module_feed_dict(features, "relative_position_encoding")
    period = relative.get("cyclic_period")
    # BioIR's Boltz-2 encoder expects periods already broadcast over pair rows.
    relative["cyclic_period"] = period[:, None] if period is not None and torch.any(period > 0) else None
    return relative


class SubsampledMSA(nn.Module):
    """Sample MSA rows with the CPU RNG on every trunk iteration."""

    def __init__(self, module: nn.Module, count: int = 1024):
        super().__init__()
        self.module = module
        self.count = count

    def forward(self, z, emb, msa, has_deletion, deletion_value, msa_paired,
                msa_mask, token_pad_mask, attn_metadata=None):
        indices = torch.randperm(msa.shape[1], device="cpu")[:self.count]
        return self.module(
            z=z, emb=emb, msa=msa[:, indices], has_deletion=has_deletion[:, indices],
            deletion_value=deletion_value[:, indices], msa_paired=msa_paired[:, indices],
            msa_mask=msa_mask[:, indices], token_pad_mask=token_pad_mask,
            attn_metadata=attn_metadata,
        )


class FixedPoseAffinity(Boltz2Affinity):
    """Score supplied poses using the Boltz-2 trunk and two affinity heads."""

    @classmethod
    def from_checkpoint(cls, checkpoint: Path):
        config = cls.get_pretrained_config()
        config.skip_run_structure = True
        config.confidence_prediction = False
        for head in (config.affinity.module1, config.affinity.module2):
            head.set_dtype(torch.float32)
        model = cls(config=config, include_load_weights=False)
        state = torch.load(checkpoint, weights_only=False, map_location="cpu")["state_dict"]
        model.load_weights(weights=state)
        model.trunk.msa_module = SubsampledMSA(model.trunk.msa_module)
        model.diffusion_conditioning = nn.Identity()
        model.diffusion_sampler = nn.Identity()
        model.confidence_module = nn.Identity()
        model.distogram_module = nn.Identity()
        return model.cpu().eval()

    @torch.inference_mode()
    def forward(self, feed_dict: dict[str, torch.Tensor], recycling_steps: int = 5):
        coordinates = fixed_coordinates(feed_dict["coords"])
        batch, atoms, _ = feed_dict["atom_to_token"].shape
        if coordinates.shape[:2] != (batch, atoms):
            raise ValueError("Fixed coordinates and feature batch disagree.")
        metadata = self.create_attn_metadata(atoms)
        s_inputs = self.input_embedder(
            **self.get_module_feed_dict(feed_dict, "input_embedder"), attn_metadata=metadata
        )
        s_init = self.s_init(s_inputs)
        z_init = self.z_init_1(s_inputs)[:, :, None] + self.z_init_2(s_inputs)[:, None, :]
        z_init = z_init + self.rel_pos(**relative_position_features(self, feed_dict))
        z_init = z_init + self.token_bonds(feed_dict["token_bonds"].float())
        if self.config.bond_type_feature:
            z_init = z_init + self.token_bonds_type(feed_dict["type_bonds"].long())
        z_init = z_init + self.contact_conditioning(
            feed_dict["contact_conditioning"], feed_dict["contact_threshold"]
        )
        # Do not enable CUDA graph capture: MSA rows are resampled on each iteration.
        _, z = self.trunk(
            s_init=s_init, z_init=z_init, s_inputs=s_inputs,
            **self.get_module_feed_dict(feed_dict, "trunk"),
            recycling_steps=recycling_steps, template_feats=None,
        )
        s_affinity = self.input_embedder(
            **self.get_module_feed_dict(feed_dict, "input_embedder_affinity"), attn_metadata=metadata
        )
        cross_mask, head_mask = create_cross_pair_mask(
            feed_dict["token_pad_mask"], feed_dict["mol_type"],
            feed_dict["affinity_token_mask"], include_mask_for_head=True,
        )
        # Match Boltzina's mask on the shared pair representation.
        z_affinity = z.float() * cross_mask[:1].float().unsqueeze(-1)
        values, probabilities = [], []
        with torch.autocast("cuda", enabled=False):
            for boundaries, module in zip(
                (self.boundaries_1, self.boundaries_2),
                (self.affinity_module1, self.affinity_module2),
            ):
                value, logits, _ = module(
                    s=s_affinity.float(), z=z_affinity,
                    distogram=compute_distogram(coordinates, boundaries, feed_dict["token_to_rep_atom"]),
                    cross_pair_mask_0=cross_mask.float(), cross_pair_mask_1=head_mask.float(),
                    attn_metadatas={},
                )
                values.append(value)
                probabilities.append(torch.sigmoid(logits))
        return {
            "affinity_pred_value": (values[0] + values[1]) / 2,
            "affinity_probability_binary": (probabilities[0] + probabilities[1]) / 2,
            "affinity_pred_value1": values[0], "affinity_pred_value2": values[1],
            "affinity_probability_binary1": probabilities[0],
            "affinity_probability_binary2": probabilities[1],
            "coords": coordinates,
        }
