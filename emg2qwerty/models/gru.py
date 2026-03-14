import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torchmetrics import MetricCollection

from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty import utils

class GRUModule(pl.LightningModule):
    def __init__(
        self,
        in_features,
        hidden_size,
        num_layers,
        dropout,
        bidirectional,
        optimizer=None,
        lr_scheduler=None,
        decoder=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.optimizer_cfg = optimizer
        self.lr_scheduler_cfg = lr_scheduler

        self.projection = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # self.gru = nn.GRU(
        #     input_size=in_features,
        #     hidden_size=hidden_size,
        #     num_layers=num_layers,
        #     dropout=dropout if num_layers > 1 else 0.0,
        #     bidirectional=bidirectional,
        # )

        out_dim = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Linear(out_dim, charset().num_classes)
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

        self._printed_shapes = False
        self.blank_idx = charset().null_class

    def forward(self, inputs):
        T = inputs.shape[0]
        N = inputs.shape[1]

        x = inputs.reshape(T, N, -1)

        if not self._printed_shapes:
            print("INPUT SHAPE:", inputs.shape)
            print("FLATTENED SHAPE:", x.shape)
            self._printed_shapes = True
            
        x = self.projection(x)
        emissions, _ = self.gru(x)
        emissions = self.classifier(emissions)
        emissions = F.log_softmax(emissions, dim=-1)
        return emissions

    def greedy_decode_batch(self, emissions, input_lengths):
        """
        emissions: (T, N, C) log-probs
        input_lengths: (N,)
        Returns a list of LabelData predictions
        """
        pred_ids = emissions.argmax(dim=-1).detach().cpu().numpy()  # (T, N)
        input_lengths = input_lengths.detach().cpu().numpy()

        predictions = []

        for i in range(pred_ids.shape[1]):
            seq = pred_ids[: input_lengths[i], i].tolist()

            collapsed = []
            prev = None
            for token in seq:
                if token != prev and token != self.blank_idx:
                    collapsed.append(token)
                prev = token

            predictions.append(LabelData.from_labels(collapsed))

        return predictions

    def _step(self, batch, stage):
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]

        emissions = self.forward(inputs)

        loss = self.ctc_loss(
            emissions,
            targets.transpose(0, 1),
            input_lengths,
            target_lengths,
        )

        batch_size = inputs.shape[1]
        self.log(f"{stage}/loss", loss, prog_bar=True, batch_size=batch_size)

        predictions = self.greedy_decode_batch(emissions, input_lengths)

        metrics = self.metrics[f"{stage}_metrics"]
        targets_np = targets.detach().cpu().numpy()
        target_lengths_np = target_lengths.detach().cpu().numpy()

        for i in range(batch_size):
            target = LabelData.from_labels(targets_np[: target_lengths_np[i], i])
            metrics.update(prediction=predictions[i], target=target)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def on_train_epoch_end(self):
        self.log_dict(self.metrics["train_metrics"].compute(), prog_bar=True)
        self.metrics["train_metrics"].reset()

    def on_validation_epoch_end(self):
        self.log_dict(self.metrics["val_metrics"].compute(), prog_bar=True)
        self.metrics["val_metrics"].reset()

    def on_test_epoch_end(self):
        self.log_dict(self.metrics["test_metrics"].compute(), prog_bar=True)
        self.metrics["test_metrics"].reset()

    def configure_optimizers(self):
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )