import os
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


def set_seed(seed: int):
    # set all seeds to ensure reproducible splits and weight initializations
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



# custom Dataset class tells PyTorch how to load and index individual data samples
class DatasetBuilder(Dataset):
    def __init__(self, source:str, target_column: str):
        df = pd.read_csv(source)
        self.target_column = target_column
        # drop target column to prevent data leakage during training
        self.features_array = df.drop(columns=[target_column]).values
        self.labels_array = df[target_column].values
        
    def __len__(self):
        # returns total number of samples in the dataset
        return len(self.features_array)

    def __getitem__(self, idx):
        # returns a single sample dictionary containing features (x) and label (y)
        raw_features = self.features_array[idx]
        raw_label = self.labels_array[idx]
        
        # convert input features to float32 tensors
        features = torch.tensor(raw_features, dtype=torch.float32)
        # classification targets must be long (int64) tensors for cross entropy loss
        label = torch.tensor(raw_label, dtype=torch.long)
        
        return {"x": features, "y": label}


# inheriting from nn.Module allows PyTorch to automatically track model parameters (weights/biases)
class Model(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        # initialize base PyTorch class to register layers
        super().__init__()
        self.output = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # defines the mathematical steps performed on the input x
        logits = self.output(x)
        return logits


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

    device: torch.device = field(default_factory=lambda: torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    ))

class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, dataset, target, config: Config):
        self.config = config
        self.device = config.device
        # move model parameters and layers to the target device memory (e.g., GPU VRAM or CPU RAM)
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        # mixed precision only helps on newer cuda devices (mps/cpu don't benefit much; act
        # probably more inefficient)
        self.use_gpu_feat = self.config.mixed_precision and self.device.type == "cuda"
        
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        full_dataset = DatasetBuilder(source=dataset, target_column=target)
        # partition dataset indices randomly into train and validation sets
        train_size = int(self.config.split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

        # pin_memory=True speeds up cpu-to-gpu transfers when using non_blocking=True, but doesnt
        # make sense to enable for cpu or mps use
        # loaders automatically group data into random mini-batches and handle multiprocessing
        self.train_loader = DataLoader(
            train_subset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=config.num_workers, 
            pin_memory=self.use_gpu_feat
        )
        self.val_loader = DataLoader(
            val_subset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=config.num_workers, 
            pin_memory=self.use_gpu_feat
        )
        
        # scale gradients to prevent underflow issues with fp16 mixed precision
        self.scaler = torch.amp.GradScaler(enabled=self.use_gpu_feat)

    def train_epoch(self) -> tuple[float, float]:
        # train mode enables layers like dropout or batch normalization to update their states
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0

        for batch in self.train_loader:
            # pin_memory in loader works with non_blocking=True to copy CPU data directly to GPU memory without CPU involvement
            x = batch["x"].to(self.device, non_blocking=True)
            y = batch["y"].to(self.device, non_blocking=True)

            # setting grads to None is faster than zeroing them out because it avoids reading the old grad data 
            # from RAM just to overwrite it with zero. The next backward pass will just allocate new tensors
            self.optimizer.zero_grad(set_to_none=True)

            # autocast runs layers in float16 for fast matrix multiplication, but keeps loss in float32 for training stability
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_gpu_feat):
                outputs = self.model(x)
                loss = self.criterion(outputs, y)

            # float16 numbers have limited range, so tiny gradients will round to zero (underflow). scaling the loss prevents this
            self.scaler.scale(loss).backward()
           
            # we must divide gradients back by the scale factor (unscale) before clipping them, otherwise 
            # the clipping threshold will be calculated on the artificially scaled values
            if self.use_gpu_feat:
                self.scaler.unscale_(self.optimizer)
            # clip gradient norms to prevent exploding gradients during updates
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # optimizer step is only called if gradients contain no NaNs/Infs (which can happen if scaling was too high)
            if self.use_gpu_feat:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # normal weight update when not using mixed precision scaling
                self.optimizer.step()

            # batch sizes can vary (e.g. the last batch is often smaller). we multiply by batch size to get a weighted sum
            running_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100
        return epoch_loss, epoch_acc

    # no_grad disables PyTorch's history tracking graph to reduce memory usage and speed up validation
    @torch.no_grad() 
    def validate(self) -> tuple[float, float]:
        # eval mode disables training-specific behaviors like dropout or running statistics updates in batch norm
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0

        for batch in self.val_loader:
            x = batch["x"].to(self.device, non_blocking=True)
            y = batch["y"].to(self.device, non_blocking=True)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_gpu_feat):
                outputs = self.model(x)
                loss = self.criterion(outputs, y)

            running_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        val_loss = running_loss / total
        val_acc = (correct / total) * 100
        return val_loss, val_acc

    def train(self):
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(1, self.config.epochs + 1):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            # step the learning rate scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            print(f"Epoch [{epoch:02d}/{self.config.epochs:02d}] | LR: {current_lr:.6f}")
            print(f"  [TRAIN] Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
            print(f"  [VALID] Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")

            # checkpoint the model if validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.config.checkpoint_dir, "best_model.pt"))
                print("  --> Saved new checkpoint.")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1


            # stop training early if validation loss hasn't improved 
            if epochs_without_improvement >= self.config.patience:
                print(f"Early stopping: no improvement for {self.config.patience} epochs.")
                break


if __name__ == "__main__":
    INPUT_FEATURES = 64
    NUM_CLASSES = 2
    config = Config()
    set_seed(config.seed) 

    model = Model(input_dim=INPUT_FEATURES, num_classes=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        dataset= ,
        target= ,
        config=config
    )
     
    trainer.train()
