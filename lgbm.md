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
