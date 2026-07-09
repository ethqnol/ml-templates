import os
import random
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from torch.utils.data import Dataset, DataLoader, random_split


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def numpy_collate(batch):
    # custom collate function to output numpy arrays and avoid converting to pytorch tensors first
    transposed = list(zip(*batch))
    x = np.stack(transposed[0])
    y = np.stack(transposed[1])
    return {"x": x, "y": y}


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
        
        # convert to float32 for model inputs
        features = raw_features.astype(np.float32)
        # convert regression target to float32
        label = np.array([raw_label], dtype=np.float32)
        
        return features, label


class Model(nnx.Module):
    def __init__(self, input_dim: int, output_dim: int, rngs: nnx.Rngs):
        self.linear = nnx.Linear(input_dim, output_dim, rngs=rngs)

    def __call__(self, x):
        # map input features directly to a single continuous output
        return self.linear(x)


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


# JAX compiles math operations into optimized machine code (XLA), which requires pure functions (no side effects)
# nnx.jit lets us write PyTorch-style stateful classes by converting them to stateless data structures (PyTrees) before compiling
# then merging the updated parameter values back into the mutable Python instances in-place.
@nnx.jit
def train_step(model: Model, optimizer: nnx.Optimizer, batch):
    # loss_fn must take the model instance as its first argument because JAX's gradient calculation (value_and_grad)
    # operates by tracking operations performed on the first argument of the function
    def loss_fn(model):
        preds = model(batch["x"])
        # optax expects integer class labels (like 0, 1, 2) rather than one-hot encoded vectors (like [1, 0, 0])
        loss = jnp.mean((preds - batch["y"]) ** 2)
        return loss, preds
    
    # nnx value_and_grad handles extracting params from the module and returns loss + gradients
    (loss, preds), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    # applies calculated gradients to update the model parameters wrapped inside the optimizer.
    optimizer.update(grads)
    
    # mean absolute error compilation
    mae = jnp.mean(jnp.abs(preds - batch["y"]))
    return loss, mae


@nnx.jit
def eval_step(model: Model, batch):
    preds = model(batch["x"])
    # mean squared error loss
    loss = jnp.mean((preds - batch["y"]) ** 2)
    
    # mean absolute error compilation
    mae = jnp.mean(jnp.abs(preds - batch["y"]))
    return loss, mae


class Trainer:
    def __init__(self, model, optimizer_fn, dataset, target, config: Config):
        self.config = config
        
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        full_dataset = DatasetBuilder(source=dataset, target_column=target)
        train_size = int(self.config.split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

        # load data to numpy arrays
        self.train_loader = DataLoader(
            train_subset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=2, 
            collate_fn=numpy_collate
        )
        self.val_loader = DataLoader(
            val_subset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=2, 
            collate_fn=numpy_collate
        )
        
        self.model = model
        self.optimizer = optimizer_fn(self.model)

    def train_epoch(self) -> tuple[float, float]:
        running_loss, running_mae, total = 0.0, 0.0, 0

        for batch in self.train_loader:
            # convert batch values to jax arrays
            batch_jax = {
                "x": jnp.array(batch["x"]),
                "y": jnp.array(batch["y"])
            }
            num_samples = batch_jax["x"].shape[0]
            
            # run training step (mutates model/optimizer state in place)
            loss, mae = train_step(self.model, self.optimizer, batch_jax)
            
            # metric compilation
            running_loss += loss.item() * num_samples
            running_mae += mae.item() * num_samples
            total += num_samples

        epoch_loss = running_loss / total
        epoch_mae = running_mae / total
        return epoch_loss, epoch_mae

    def validate(self) -> tuple[float, float]:
        running_loss, running_mae, total = 0.0, 0.0, 0

        for batch in self.val_loader:
            # convert batch values to jax arrays
            batch_jax = {
                "x": jnp.array(batch["x"]),
                "y": jnp.array(batch["y"])
            }
            num_samples = batch_jax["x"].shape[0]
            
            # run evaluation step
            loss, mae = eval_step(self.model, batch_jax)
            
            # metric compilation
            running_loss += loss.item() * num_samples
            running_mae += mae.item() * num_samples
            total += num_samples

        val_loss = running_loss / total
        val_mae = running_mae / total
        return val_loss, val_mae

    def train(self):
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(1, self.config.epochs + 1):
            train_loss, train_mae = self.train_epoch()
            val_loss, val_mae = self.validate()

            print(f"Epoch [{epoch:02d}/{self.config.epochs:02d}]")
            print(f"  [TRAIN] MSE Loss: {train_loss:.4f} | MAE: {train_mae:.4f}")
            print(f"  [VALID] MSE Loss: {val_loss:.4f} | MAE: {val_mae:.4f}")

            # checkpoint the model if validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # extract model state for saving
                state_dict = nnx.state(self.model)
                with open(os.path.join(self.config.checkpoint_dir, "best_model.pkl"), "wb") as f:
                    pickle.dump(state_dict, f)
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
    OUTPUT_DIM = 1
    config = Config()
    set_seed(config.seed) 

    # initialize rngs
    rngs = nnx.Rngs(config.seed)
    model = Model(input_dim=INPUT_FEATURES, output_dim=OUTPUT_DIM, rngs=rngs)
    
    def optimizer_fn(m):
        return nnx.Optimizer(m, optax.adamw(config.learning_rate, config.weight_decay))

    trainer = Trainer(
        model=model,
        optimizer_fn=optimizer_fn,
        dataset= ,
        target= ,
        config=config
    )
     
    trainer.train()
