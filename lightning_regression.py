import os
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DatasetBuilder(Dataset):
    def __init__(self, source: str, target_column: str):
        df = pd.read_csv(source)
        self.target_column = target_column
        self.features_array = df.drop(columns=[target_column]).values
        self.labels_array = df[target_column].values
        
    def __len__(self):
        return len(self.features_array)

    def __getitem__(self, idx):
        raw_features = self.features_array[idx]
        raw_label = self.labels_array[idx]
        
        # convert to float32
        features = torch.tensor(raw_features, dtype=torch.float32)
        # unsqueeze regression target so its shape (batch_size, 1) matches regression loss dimensions correctly
        label = torch.tensor(raw_label, dtype=torch.float32).unsqueeze(-1)
        
        return {"x": features, "y": label}


class RegressorModule(pl.LightningModule):
    def __init__(self, input_dim: int, output_dim: int, lr: float, weight_decay: float):
        super().__init__()
        self.save_hyperparameters()
        self.output = nn.Linear(input_dim, output_dim)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.output(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        preds = self(x)
        loss = self.criterion(preds, y)
        
        # calculate mean absolute error
        mae = torch.abs(preds - y).mean()
        
        # logging on epoch avoids step-level log noise in tensorboard/consoles
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_mae", mae, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        preds = self(x)
        loss = self.criterion(preds, y)
        
        # calculate mean absolute error
        mae = torch.abs(preds - y).mean()
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mae", mae, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        # self.trainer is automatically injected by lightning, making max_epochs accessible here
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }


@dataclass
class Config:
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    split: float = 0.8 
    num_workers: int = 2
    seed: int = 42
    checkpoint_dir: str = "./checkpoints"
    mixed_precision: bool = True 


class Trainer:
    def __init__(self, model, dataset, target, config: Config):
        self.config = config
        
        # build dataset and split
        full_dataset = DatasetBuilder(source=dataset, target_column=target)
        train_size = int(self.config.split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

        # build dataloaders
        self.train_loader = DataLoader(
            train_subset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            val_subset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

        self.model = model

    def train(self):
        # setup callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.checkpoint_dir,
            filename="best_model",
            monitor="val_loss",
            mode="min",
            save_top_k=1
        )
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=self.config.patience,
            mode="min"
        )

        # initialize trainer
        trainer = pl.Trainer(
            max_epochs=self.config.epochs,
            callbacks=[checkpoint_callback, early_stop_callback],
            # precision="16-mixed" runs FP16 mixed precision for faster training on cuda GPUs
            precision="16-mixed" if self.config.mixed_precision else "32",
            accelerator="auto",
            devices=1,
            enable_progress_bar=True
        )

        # train model
        trainer.fit(
            self.model,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader
        )


if __name__ == "__main__":
    INPUT_FEATURES = 64
    OUTPUT_DIM = 1
    config = Config()
    set_seed(config.seed) 

    model = RegressorModule(
        input_dim=INPUT_FEATURES,
        output_dim=OUTPUT_DIM,
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    trainer = Trainer(
        model=model,
        dataset= ,
        target= ,
        config=config
    )
     
    trainer.train()
