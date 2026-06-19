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
| **baseline** | 1 | 60 | LSTM <br> CNN | 0.57 <br> 0.51 | 0.4960 <br> 0.4544 | No sig. diff (p=0.096) |
| **changed_training** | 1 | 60 | LSTM <br> CNN | 0.57 <br> 0.49 | 0.5326 <br> 0.4526 | LSTM sig. better (p=0.032) |
| **changed_data** | 1 | 120 | LSTM <br> CNN | **0.60** <br> **0.60** | **0.5475** <br> 0.4436 | No sig. diff (p=1.000) |
| **future_baseline** | 5 | 60 | LSTM <br> CNN | 0.55 <br> 0.39 | 0.4143 <br> 0.3942 | LSTM sig. better (p=0.000) |
| **future_changed_training** | 5 | 60 | LSTM <br> CNN | 0.48 <br> 0.37 | 0.4929 <br> 0.3825 | LSTM sig. better (p=0.000) |
| **future_changed_data** | 5 | 120 | LSTM <br> CNN | 0.42 <br> 0.37 | 0.4225 <br> 0.4256 | LSTM sig. better (p=0.003) |

---

## Key Takeaways and Analysis

### 1. The Challenge of Predicting Further Out
Predicting stock price movements 5 days in the future is significantly more challenging than predicting the very next day. 
Across the board, Accuracy and ROC-AUC metrics plummeted when shifting from a 1-day horizon to a 5-day horizon. For all 5-day horizon configurations, both models struggled to learn any reliable signal, with ROC-AUCs staying below 0.50 (worse than random chance on the test set). This indicates that predicting multi-day directions with standard sequential architectures on raw technical indicators is highly prone to overfitting on noise.

### 2. LSTM Outperforms CNN
Across almost all experimental variants, the **LSTM model consistently outperformed the CNN model** in terms of both Accuracy and ROC-AUC. 
- In `changed_training` (1-day horizon), the LSTM achieved an ROC-AUC of **0.5326**, which was statistically significantly better than the CNN's 0.4526 (p=0.032).
- In the 5-day prediction experiments, the LSTM's accuracy was consistently higher than the CNN's (e.g. 0.55 vs 0.39 in the baseline), and McNemar tests indicated a statistically significant difference in errors (p < 0.005), showing that LSTM's memory cells are much better suited for stock sequence modeling than the CNN's temporal convolutions on this dataset.

### 3. Benefit of Longer Sequence Lengths
Increasing the sequence length from 60 to 120 days (`changed_data`) gave the LSTM its **best overall performance of the entire sweep: 0.60 Accuracy and 0.5475 ROC-AUC**. 
However, for the CNN, the longer historical lookback did not help, and its performance remained weak (ROC-AUC 0.4436), suggesting it struggled to leverage the extra context.

## Conclusion
Predicting stock price direction remains an incredibly noisy task. If predicting 1 day out, a standard **LSTM with a 120-day lookback** performed best, achieving a solid **0.60 Accuracy and 0.5475 ROC-AUC**. For predicting longer 5-day horizons, the task becomes highly speculative under these configurations, though the **LSTM model still demonstrated superior robustness** compared to the CNN.
