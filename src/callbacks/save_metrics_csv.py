# callbacks/save_metrics_csv.py
from pathlib import Path
import csv, torch
from pytorch_lightning.callbacks import Callback

class SaveMetricsCSV(Callback):
    """
    Writes one CSV per run:
        epoch, metric_1, metric_2, ...
    """
    def __init__(self, run_dir: str = "metrics"):
        super().__init__()
        self.run_dir = Path(run_dir)
        self.header_written = False         # <- ensure it ALWAYS exists
        self.fcsv = None
        self.writer = None

    # ---------------------------------------------------------------------
    def on_train_start(self, trainer, pl_module):
        # only open the file once real training begins
        self.fpath = (self.run_dir / "epoch_metrics.csv").resolve()
        self.fpath.parent.mkdir(parents=True, exist_ok=True)
        self.fcsv  = open(self.fpath, "w", newline="")
        self.writer = csv.writer(self.fcsv)

    # ---------------------------------------------------------------------
    def _scalar(self, x):
        return x.item() if torch.is_tensor(x) else x

    def on_validation_epoch_end(self, trainer, pl_module):
        # skip the sanity-check loop
        if trainer.sanity_checking:
            return

        metrics = trainer.callback_metrics
        if not self.header_written:
            self.writer.writerow(["epoch", *metrics.keys()])
            self.header_written = True

        row = [trainer.current_epoch,
               *[self._scalar(metrics[k]) for k in metrics]]
        self.writer.writerow(row)
        self.fcsv.flush()

    def on_train_end(self, trainer, pl_module):
        if self.fcsv is not None:
            self.fcsv.close()
            pl_module.print(f"[SaveMetricsCSV] wrote {self.fpath}")
