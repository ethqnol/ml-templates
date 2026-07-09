import os
import random
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


@dataclass
class Config:
    n_trials: int = 20
    test_size: float = 0.2
    val_size: float = 0.2
    seed: int = 42
    checkpoint_dir: str = "./checkpoints"


class Trainer:
    def __init__(self, dataset, target, config: Config):
        self.config = config
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        # load dataset
        df = pd.read_csv(dataset)
        X = df.drop(columns=[target]).values
        y = df[target].values
        
        # split data into train, validation, and test sets
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.seed
        )
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=self.config.val_size, random_state=self.config.seed
        )

    def optimize_and_train(self):
        # objective function for optuna
        def objective(trial):
            params = {
                "objective": "regression",
                "boosting_type": "gbdt",
                "random_state": self.config.seed,
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "verbosity": -1,
            }
            
            # evaluate parameters on validation set
            model = lgb.LGBMRegressor(**params)
            model.fit(
                self.X_train, 
                self.y_train, 
                eval_set=[(self.X_val, self.y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
            )
            
            preds = model.predict(self.X_val)
            rmse = np.sqrt(mean_squared_error(self.y_val, preds))
            return rmse

        # optimize hyperparameters with optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.config.n_trials)
        best_params = study.best_params
        
        print(f"best validation rmse: {study.best_value:.4f}")
        print("parameters:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
        # retrain on full training data (train + val) using the best params found
        X_train_full = np.concatenate([self.X_train, self.X_val], axis=0)
        y_train_full = np.concatenate([self.y_train, self.y_val], axis=0)
        
        final_params = {
            "objective": "regression",
            "boosting_type": "gbdt",
            "random_state": self.config.seed,
            "verbosity": -1,
            **best_params
        }
        
        self.best_model = lgb.LGBMRegressor(**final_params)
        self.best_model.fit(X_train_full, y_train_full)
        
        # evaluate best model
        test_preds = self.best_model.predict(self.X_test)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, test_preds))
        print(f"test rmse: {test_rmse:.4f}")
        
        # save checkpoint
        checkpoint_path = os.path.join(self.config.checkpoint_dir, "best_lgbm_model.pkl")
        with open(checkpoint_path, "wb") as f:
            pickle.dump(self.best_model, f)


if __name__ == "__main__":
    config = Config()
    set_seed(config.seed)

    trainer = Trainer(
        dataset= ,
        target= ,
        config=config
    )

    trainer.optimize_and_train()
