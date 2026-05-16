import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from boltz.data import const
from boltz.data.crop.affinity import AffinityCropper
from boltz.data.feature.featurizerv2 import Boltz2Featurizer
from boltz.data.mol import load_canonicals, load_molecules
from boltz.data.pad import pad_to_max
from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
from boltz.data.types import (
    MSA,
    Input,
    Manifest,
    Record,
    ResidueConstraints,
    StructureV2,
)


def load_input(
    record: Record,
    target_dir: Path,
    msa_dir: Path,
    constraints_dir: Optional[Path] = None,
    template_dir: Optional[Path] = None,
    extra_mols_dir: Optional[Path] = None,
    affinity: bool = False,
) -> Input:
    """Load the given input data.

    Parameters
    ----------
    record : Record
        The record to load.
    target_dir : Path
        The path to the data directory.
    msa_dir : Path
        The path to msa directory.
    constraints_dir : Optional[Path]
        The path to the constraints directory.
    template_dir : Optional[Path]
        The path to the template directory.
    extra_mols_dir : Optional[Path]
        The path to the extra molecules directory.
    affinity : bool
        Whether to load the affinity data.

    Returns
    -------
    Input
        The loaded input.

    """
    # Load the structure
    if affinity:
        structure = StructureV2.load(
            target_dir / record.id / f"pre_affinity_{record.id}.npz"
        )
    else:
        structure = StructureV2.load(target_dir / f"{record.id}.npz")

    msas = {}
    for chain in record.chains:
        msa_id = chain.msa_id
        # Load the MSA for this chain, if any
        if msa_id != -1:
            msa = MSA.load(msa_dir / f"{msa_id}.npz")
            msas[chain.chain_id] = msa

    # Load templates
    templates = None
    if record.templates and template_dir is not None:
        templates = {}
        for template_info in record.templates:
            template_id = template_info.name
            template_path = template_dir / f"{record.id}_{template_id}.npz"
            template = StructureV2.load(template_path)
            templates[template_id] = template

    # Load residue constraints
    residue_constraints = None
    if constraints_dir is not None:
        residue_constraints = ResidueConstraints.load(
            constraints_dir / f"{record.id}.npz"
        )

    # Load extra molecules
    extra_mols = {}
    if extra_mols_dir is not None:
        extra_mol_path = extra_mols_dir / f"{record.id}.pkl"
        if extra_mol_path.exists():
            with extra_mol_path.open("rb") as f:
                extra_mols = pickle.load(f)  # noqa: S301

    return Input(
        structure,
        msas,
        record=record,
        residue_constraints=residue_constraints,
        templates=templates,
        extra_mols=extra_mols,
    )


def collate(data: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate the data.

    Parameters
    ----------
    data : List[Dict[str, Tensor]]
        The data to collate.

    Returns
    -------
    Dict[str, Tensor]
        The collated data.

    """
    # Get the keys
    keys = data[0].keys()

    # Collate the data
    collated = {}
    for key in keys:
        values = [d[key] for d in data]

        if key not in [
            "all_coords",
            "all_resolved_mask",
            "crop_to_all_atom_map",
            "chain_symmetries",
            "amino_acids_symmetries",
            "ligand_symmetries",
            "record",
            "affinity_mw",
        ]:
            # Check if all have the same shape
            shape = values[0].shape
            if not all(v.shape == shape for v in values):
                values, _ = pad_to_max(values, 0)
            else:
                values = torch.stack(values, dim=0)

        # Stack the values
        collated[key] = values

    return collated


class PredictionDataset(torch.utils.data.Dataset):
    """Base iterable dataset."""

    def __init__(
        self,
        manifest: Manifest,
        target_dir: Path,
        msa_dir: Path,
        mol_dir: Path,
        constraints_dir: Optional[Path] = None,
        template_dir: Optional[Path] = None,
        extra_mols_dir: Optional[Path] = None,
        override_method: Optional[str] = None,
        affinity: bool = False,
    ) -> None:
        """Initialize the training dataset.

        Parameters
        ----------
        manifest : Manifest
            The manifest to load data from.
        target_dir : Path
            The path to the target directory.
        msa_dir : Path
            The path to the msa directory.
        mol_dir : Path
            The path to the moldir.
        constraints_dir : Optional[Path]
            The path to the constraints directory.
        template_dir : Optional[Path]
            The path to the template directory.

        """
        super().__init__()
        self.manifest = manifest
        self.target_dir = target_dir
        self.msa_dir = msa_dir
        self.mol_dir = mol_dir
        self.constraints_dir = constraints_dir
        self.template_dir = template_dir
        self.tokenizer = Boltz2Tokenizer()
        self.featurizer = Boltz2Featurizer()
        self.canonicals = load_canonicals(self.mol_dir)
        self.extra_mols_dir = extra_mols_dir
        self.override_method = override_method
        self.affinity = affinity
        if self.affinity:
            self.cropper = AffinityCropper()
        self.failed_records: list[dict[str, str]] = []
        self.valid_indices = list(range(len(self.manifest.records)))
        if self.affinity:
            self.valid_indices = self._prefilter_affinity_records()

    def _record_error(self, record: Record, stage: str, error: Exception) -> RuntimeError:
        return RuntimeError(f"{stage} failed on {record.id}: {error}")

    def _load_and_crop(self, record: Record) -> tuple[Input, object]:
        try:
            input_data = load_input(
                record=record,
                target_dir=self.target_dir,
                msa_dir=self.msa_dir,
                constraints_dir=self.constraints_dir,
                template_dir=self.template_dir,
                extra_mols_dir=self.extra_mols_dir,
                affinity=self.affinity,
            )
        except Exception as e:  # noqa: BLE001
            raise self._record_error(record, "Input loading", e) from e

        try:
            tokenized = self.tokenizer.tokenize(input_data)
        except Exception as e:  # noqa: BLE001
            raise self._record_error(record, "Tokenizer", e) from e

        if self.affinity:
            try:
                tokenized = self.cropper.crop(
                    tokenized,
                    max_tokens=256,
                    max_atoms=2048,
                )
            except Exception as e:  # noqa: BLE001
                raise self._record_error(record, "Affinity cropper", e) from e

        return input_data, tokenized

    def _prefilter_affinity_records(self) -> list[int]:
        valid_indices = []
        for idx, record in enumerate(self.manifest.records):
            try:
                self._load_and_crop(record)
            except RuntimeError as e:
                print(f"{e}. Skipping record.")  # noqa: T201
                self.failed_records.append({"id": record.id, "error": str(e)})
                continue
            valid_indices.append(idx)

        if self.failed_records:
            failed_ids = ", ".join(item["id"] for item in self.failed_records[:10])
            if len(self.failed_records) > 10:
                failed_ids += ", ..."
            print(  # noqa: T201
                f"Skipping {len(self.failed_records)} invalid affinity record(s): {failed_ids}"
            )
        return valid_indices

    def __getitem__(self, idx: int) -> dict:
        """Get an item from the dataset.

        Returns
        -------
        Dict[str, Tensor]
            The sampled data features.

        """
        # Get record
        record = self.manifest.records[self.valid_indices[idx]]
        input_data, tokenized = self._load_and_crop(record)

        # Load conformers
        try:
            molecules = {}
            molecules.update(self.canonicals)
            molecules.update(input_data.extra_mols)
            mol_names = set(tokenized.tokens["res_name"].tolist())
            mol_names = mol_names - set(molecules.keys())
            molecules.update(load_molecules(self.mol_dir, mol_names))
        except Exception as e:  # noqa: BLE001
            raise self._record_error(record, "Molecule loading", e) from e

        # Inference specific options
        options = record.inference_options
        if options is None:
            pocket_constraints, contact_constraints = None, None
        else:
            pocket_constraints, contact_constraints = (
                options.pocket_constraints,
                options.contact_constraints,
            )

        # Get random seed
        seed = 42
        random = np.random.default_rng(seed)

        # Compute features
        try:
            features = self.featurizer.process(
                tokenized,
                molecules=molecules,
                random=random,
                training=False,
                max_atoms=None,
                max_tokens=None,
                max_seqs=const.max_msa_seqs,
                pad_to_max_seqs=False,
                single_sequence_prop=0.0,
                compute_frames=True,
                inference_pocket_constraints=pocket_constraints,
                inference_contact_constraints=contact_constraints,
                compute_constraint_features=True,
                override_method=self.override_method,
                compute_affinity=self.affinity,
            )
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            raise self._record_error(record, "Featurizer", e) from e

        # Add record
        features["record"] = record
        return features

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        return len(self.valid_indices)


class Boltz2InferenceDataModule(pl.LightningDataModule):
    """DataModule for Boltz2 inference."""

    def __init__(
        self,
        manifest: Manifest,
        target_dir: Path,
        msa_dir: Path,
        mol_dir: Path,
        num_workers: int,
        constraints_dir: Optional[Path] = None,
        template_dir: Optional[Path] = None,
        extra_mols_dir: Optional[Path] = None,
        override_method: Optional[str] = None,
        affinity: bool = False,
        batch_size: int = 1,
    ) -> None:
        """Initialize the DataModule.

        Parameters
        ----------
        manifest : Manifest
            The manifest to load data from.
        target_dir : Path
            The path to the target directory.
        msa_dir : Path
            The path to the msa directory.
        mol_dir : Path
            The path to the moldir.
        num_workers : int
            The number of workers to use.
        constraints_dir : Optional[Path]
            The path to the constraints directory.
        template_dir : Optional[Path]
            The path to the template directory.
        extra_mols_dir : Optional[Path]
            The path to the extra molecules directory.
        override_method : Optional[str]
            The method to override.

        """
        super().__init__()
        self.num_workers = num_workers
        self.manifest = manifest
        self.target_dir = target_dir
        self.msa_dir = msa_dir
        self.mol_dir = mol_dir
        self.constraints_dir = constraints_dir
        self.template_dir = template_dir
        self.extra_mols_dir = extra_mols_dir
        self.override_method = override_method
        self.affinity = affinity
        self.batch_size = batch_size
        self._predict_dataset: Optional[PredictionDataset] = None

    def predict_dataset(self) -> PredictionDataset:
        if self._predict_dataset is None:
            self._predict_dataset = PredictionDataset(
                manifest=self.manifest,
                target_dir=self.target_dir,
                msa_dir=self.msa_dir,
                mol_dir=self.mol_dir,
                constraints_dir=self.constraints_dir,
                template_dir=self.template_dir,
                extra_mols_dir=self.extra_mols_dir,
                override_method=self.override_method,
                affinity=self.affinity,
            )
        return self._predict_dataset

    def predict_dataloader(self) -> DataLoader:
        """Get the training dataloader.

        Returns
        -------
        DataLoader
            The training dataloader.

        """
        dataset = self.predict_dataset()
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=collate,
        )

    def transfer_batch_to_device(
        self,
        batch: dict,
        device: torch.device,
        dataloader_idx: int,  # noqa: ARG002
    ) -> dict:
        """Transfer a batch to the given device.

        Parameters
        ----------
        batch : Dict
            The batch to transfer.
        device : torch.device
            The device to transfer to.
        dataloader_idx : int
            The dataloader index.

        Returns
        -------
        np.Any
            The transferred batch.

        """
        for key in batch:
            if key not in [
                "all_coords",
                "all_resolved_mask",
                "crop_to_all_atom_map",
                "chain_symmetries",
                "amino_acids_symmetries",
                "ligand_symmetries",
                "record",
                "affinity_mw",
            ]:
                batch[key] = batch[key].to(device)
        return batch
