import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig
from typing import Any
from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.data import LabelData
from torchmetrics import MetricCollection
from hydra.utils import instantiate

class EMGLSTMModule(pl.LightningModule):
    def __init__(
        self, 
        input_channels: int, 
        hidden_size: int, 
        num_layers: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.projection = nn.Linear(input_channels, 128)
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, charset().num_classes)
        )

        self.ctc_loss = nn.CTCLoss(blank=charset().null_class, zero_infinity=True)

        self.decoder = instantiate(decoder)

        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict({
            f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
            for phase in ["train", "val", "test"]
        })

    def forward(self, x):
        T, N = x.shape[0], x.shape[1]
        x = x.reshape(T, N, -1) 
        x = torch.relu(self.projection(x))
        lstm_out, _ = self.lstm(x)
        logits = self.classifier(lstm_out)
        return logits.log_softmax(2)

    def _shared_step(self, phase, batch):
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]

        emissions = self(inputs)
        
        loss = self.ctc_loss(
            log_probs=emissions, 
            targets=targets.transpose(0, 1),
            input_lengths=input_lengths, 
            target_lengths=target_lengths
        )
        
        self.log(f"{phase}/loss", loss, sync_dist=True, batch_size=inputs.shape[1])

        if phase != "train":
          metrics = self.metrics[f"{phase}_metrics"]
          
          predictions = self.decoder.decode_batch(
              emissions=emissions.detach().cpu().numpy(),
              emission_lengths=input_lengths.detach().cpu().numpy()
          )
        
          targets_np = targets.detach().cpu().numpy()
          target_lengths_np = target_lengths.detach().cpu().numpy()
        
          for i in range(len(predictions)):
              actual_target_labels = targets_np[:target_lengths_np[i], i]
              target_label_data = LabelData.from_labels(actual_target_labels)
              
              metrics.update(prediction=predictions[i], target=target_label_data)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step("train", batch)

    def validation_step(self, batch, batch_idx):
        return self._shared_step("val", batch)

    def test_step(self, batch, batch_idx):
        return self._shared_step("test", batch)

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )