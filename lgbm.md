# LightGBM + Optuna Training Loop


## LightGBM + Optuna Loop

For tabular models, this template integrates hyperparameter optimization using **Optuna** with final model retraining.

### Execution Lifecycle
```
                       +-------------------------+
                       |  Split train/val/test   |
                       +------------+------------+
                                    |
                                    v
                       +-------------------------+
                       | Run Optuna Study trials |
                       | - Train on partition    |
                       | - Evaluate on val       |
                       +------------+------------+
                                    |
                                    v
                       +-------------------------+
                       | Extract best parameters |
                       +------------+------------+
                                    |
                                    v
                       +-------------------------+
                       | Concatenate train + val |
                       | & Retrain final model   |
                       +------------+------------+
                                    |
                                    v
                       +-------------------------+
                       | Evaluate on holdout test|
                       | & pickle model save     |
                       +-------------------------+
```


> **Data Leakage in Optimization:** Optuna must only search hyperparameters using evaluation metrics computed on the separate validation partition. Once search completes, we concatenate the training and validation sets to fit the final tree estimator. This maximizes data utilization prior to final test evaluation.

---

## LightGBM + Optuna Customization Parameters

To customize LightGBM/Optuna templates for different tasks, modify the values inside the `__main__` entrypoint block.

### LightGBM Classification Customization

In [lgbm_classification.py](lgbm_classification.py#L116-L127), configure the dataset file path and target label column:

```python
if __name__ == "__main__":
    config = Config()
    set_seed(config.seed)

    trainer = Trainer(
        dataset= ,  # Set to dataset CSV path (e.g. "dataset.csv")
        target= ,   # Set to target column name string
        config=config
    )

    trainer.optimize_and_train()
```

### LightGBM Regression Customization

In [lgbm_regression.py](lgbm_regression.py#L111-L122), set the dataset file path and target label column:

```python
if __name__ == "__main__":
    config = Config()
    set_seed(config.seed)

    trainer = Trainer(
        dataset= ,  # Set to dataset CSV path (e.g. "dataset.csv")
        target= ,   # Set to target column name string
        config=config
    )

    trainer.optimize_and_train()
```

---

## LightGBM + Optuna Customization Options

You can customize the hyperparameter search boundaries, early stopping limits, and training configurations depending on the task.

### Optuna Search boundaries

To search a different range of hyperparameters, update the distributions defined in the inner `objective` function of `optimize_and_train()` (`lgbm_classification.py` lines 49-62 and `lgbm_regression.py` lines 48-60):

```python
        def objective(trial):
            params = {
                "objective": "regression",
                "boosting_type": "gbdt",
                "random_state": self.config.seed,
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.5, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 31, 255),
                "max_depth": trial.suggest_int("max_depth", 4, 12),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "verbosity": -1,
            }
```

### Early Stopping

Early stopping halts fitting if tree estimators stop improving on validation targets. Customize this inside the `model.fit` call boundaries:
* `lgbm_classification.py` line 70.
* `lgbm_regression.py` line 68.
  ```python
  callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
  ```

### Objectives and Metrics

Modify the `"objective"` and evaluation metric calculation steps:

* **LightGBM Classification**:
  * Set objective to `"binary"` for binary targets, or `"multiclass"` for multi-label targets (`lgbm_classification.py` lines 50 & 95).
* **LightGBM Regression**:
  * Set objective to `"regression"` (L2 loss) or change to `"regression_l1"` (MAE loss) or `"huber"` (Huber loss) (`lgbm_regression.py` lines 49 & 90).

### Configurations

Change the number of optimization trials and database train/validation split percentages inside the `Config` dataclass:
* `lgbm_classification.py` (lines 18-25).
* `lgbm_regression.py` (lines 18-25).
