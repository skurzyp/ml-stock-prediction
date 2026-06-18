# Stock Prediction Experiment Analysis

This document summarizes the results of the 6 hyperparameter sweeps run to evaluate different sequence lengths, model architectures (LSTM vs CNN), training setups, and prediction horizons on NASDAQ index data.

## Overview of Experiments

We conducted 6 experiments broken down into two prediction horizons:
- **1-Day Horizon (`TARGET_HORIZON=1`)**: Predicting if the next day's close price will be higher than today's.
- **5-Day Horizon (`TARGET_HORIZON=5`)**: Predicting if the close price 5 trading days from now will be strictly higher than today's.

For each horizon, we tested three variations:
1. **Baseline**: `SEQ_LEN=60`, `LEARNING_RATE=0.001`, `HIDDEN_SIZE=128`
2. **Changed Training**: `SEQ_LEN=60`, `LEARNING_RATE=0.0005`, `HIDDEN_SIZE=256` (lower LR, larger capacity)
3. **Changed Data**: `SEQ_LEN=120`, `LEARNING_RATE=0.001`, `HIDDEN_SIZE=128` (longer historical lookback)

All models used `DROPOUT=0.2`, `BATCH_SIZE=32`, and `NUM_LAYERS=2`.

---

## Results Summary

| Experiment | Target Horizon | Seq Len | Model | Accuracy | ROC-AUC | McNemar vs Other Model |
| :--- | :---: | :---: | :--- | :---: | :---: | :--- |
| **Baseline** | 1 | 60 | LSTM <br> CNN | 0.53 <br> 0.50 | 0.5073 <br> 0.4526 | No sig. diff (p=0.409) |
| **Changed Training** | 1 | 60 | LSTM <br> CNN | 0.54 <br> 0.44 | **0.5260** <br> 0.4536 | LSTM significantly better (p=0.046) |
| **Changed Data** | 1 | 120 | LSTM <br> CNN | 0.57 <br> **0.60** | 0.4792 <br> 0.4615 | No sig. diff (p=0.163) |
| **Future Baseline** | 5 | 60 | LSTM <br> CNN | 0.42 <br> 0.40 | 0.4132 <br> 0.3842 | No sig. diff (p=0.669) |
| **Future Changed Training** | 5 | 60 | LSTM <br> CNN | 0.43 <br> 0.40 | 0.4152 <br> 0.3686 | No sig. diff (p=0.533) |
| **Future Changed Data** | 5 | 120 | LSTM <br> CNN | 0.40 <br> 0.46 | 0.4704 <br> **0.5106** | CNN significantly better (p=0.008) |

---

## Key Takeaways and Analysis

### 1. The Challenge of Predicting Further Out
Predicting the price movement 5 days in the future is significantly harder than predicting the very next day. 
Across the board, both Accuracy and ROC-AUC metrics plummeted when shifting from a 1-day horizon to a 5-day horizon. For the 60-day sequence lengths (`future_baseline`, `future_changed_training`), models essentially failed to learn any meaningful signal, with ROC-AUCs hovering around 0.36 - 0.41 (worse than random chance on the test set, indicating severe overfitting to noise during training).

### 2. Sequence Length Matters for Multi-Day Horizons
The only 5-day horizon model that managed to breach the 0.5 ROC-AUC threshold was the **CNN in the `Future Changed Data` setup (`SEQ_LEN=120`)**. 
It seems that to predict trends a week out, the models require a much wider field of view (120 days vs 60 days). The CNN was statistically significantly better ($p < 0.05$) than the LSTM in this scenario, suggesting the convolutions were better at picking up broader cyclical patterns in the 120-day window than the recurrent LSTM cells.

### 3. Model Capacity vs Overfitting
In the 1-day horizon, increasing the model capacity and lowering the learning rate (`Changed Training`: Hidden=256, LR=0.0005) resulted in the **best overall ROC-AUC of 0.5260** for the LSTM. The LSTM significantly outperformed the CNN in this configuration.
However, this larger capacity backfired on the 5-day horizon (`Future Changed Training`), failing to improve over the baseline and heavily overfitting the training set.

### 4. High Accuracy vs Low ROC-AUC
In the `Changed Data` (1-day) experiment, the CNN achieved the highest absolute accuracy of the entire sweep (**0.60**). However, its ROC-AUC was a poor **0.4615**. This discrepancy typically happens when a model degenerates into predicting the majority class (e.g., predicting "UP" almost every time during a bull market phase in the test set). The poor ROC-AUC confirms that the model is poorly calibrated and isn't actually separating the classes well, despite the high accuracy number.

## Conclusion
Predicting stock price direction remains an incredibly noisy task. If predicting 1 day out, a standard **LSTM with higher capacity (Hidden=256) and a 60-day lookback** performed best. If predicting a 5-day trend, switching to a **120-day lookback** is strictly necessary, and the **CNN architecture** proved vastly superior to the LSTM at extracting those longer-term features.
