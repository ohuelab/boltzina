from pytorch_lightning.callbacks import BasePredictionWriter
from pathlib import Path
import json
from torch import Tensor
from pytorch_lightning import Trainer, LightningModule

class BoltzAffinityWriter(BasePredictionWriter):
    """Custom writer for predictions."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
    ) -> None:
        """Initialize the writer.

        Parameters
        ----------
        output_dir : str
            The directory to save the predictions.

        """
        super().__init__(write_interval="batch")
        self.failed = 0
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
        prediction: dict[str, Tensor],
        batch_indices: list[int],  # noqa: ARG002
        batch: dict[str, Tensor],
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int,  # noqa: ARG002
    ) -> None:
        """Write the predictions to disk."""
        if prediction["exception"]:
            self.failed += 1
            return

        # Handle batch size > 1
        batch_size = len(batch["record"])

        for i in range(batch_size):
            # Dump affinity summary for each item in batch
            affinity_summary = {}
            pred_affinity_value = prediction["affinity_pred_value"][i]
            pred_affinity_probability = prediction["affinity_probability_binary"][i]
            affinity_summary = {
                "affinity_pred_value": pred_affinity_value.item(),
                "affinity_probability_binary": pred_affinity_probability.item(),
            }
            if "affinity_pred_value1" in prediction:
                pred_affinity_value1 = prediction["affinity_pred_value1"][i]
                pred_affinity_probability1 = prediction["affinity_probability_binary1"][i]
                pred_affinity_value2 = prediction["affinity_pred_value2"][i]
                pred_affinity_probability2 = prediction["affinity_probability_binary2"][i]
                affinity_summary["affinity_pred_value1"] = pred_affinity_value1.item()
                affinity_summary["affinity_probability_binary1"] = (
                    pred_affinity_probability1.item()
                )
                affinity_summary["affinity_pred_value2"] = pred_affinity_value2.item()
                affinity_summary["affinity_probability_binary2"] = (
                    pred_affinity_probability2.item()
                )

            # Save the affinity summary
            struct_dir = self.output_dir / batch["record"][i].id
            struct_dir.mkdir(exist_ok=True)
            path = struct_dir / f"affinity_{batch['record'][i].id}.json"

            with path.open("w") as f:
                f.write(json.dumps(affinity_summary, indent=4))

    def on_predict_epoch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        """Print the number of failed examples."""
        # Print number of failed examples
        print(f"Number of failed examples: {self.failed}")  # noqa: T201
