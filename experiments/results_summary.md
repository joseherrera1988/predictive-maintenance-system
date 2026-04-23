# CMAPSS Benchmark Results — RF vs XGBoost vs LSTM

Failure threshold: RUL ≤ 30 cycles. All models evaluated on NASA CMAPSS FD001–FD004.

## Recall 

| Dataset | RF     | XGBoost | LSTM   |
|---------|--------|---------|--------|
| FD001   | 0.68   | 0.72    | **0.80** |
| FD002   | 0.90   | 0.93    | **0.97** |
| FD003   | 0.75   | 0.65    | **0.95** |
| FD004   | 0.74   | **0.75** | 0.66   |

## Precision

| Dataset | RF     | XGBoost | LSTM   |
|---------|--------|---------|--------|
| FD001   | 0.944  | **0.947** | 0.909 |
| FD002   | 0.965  | 0.934   | **0.967** |
| FD003   | 0.938  | 0.929   | **0.950** |
| FD004   | **0.929** | 0.909 | 0.854  |

## ROC-AUC

| Dataset | RF     | XGBoost | LSTM   |
|---------|--------|---------|--------|
| FD001   | 0.974  | 0.970   | **0.990** |
| FD002   | 0.990  | 0.988   | **0.998** |
| FD003   | 0.989  | 0.985   | **0.995** |
| FD004   | **0.979** | **0.979** | 0.977 |

## Summary
- LSTM leads on FD001–FD003 across all three metrics
- FD004 (6 operating conditions + 2 fault modes) is the hardest dataset — LSTM underperforms there, identified as the priority target for optimization
- RF is the most consistent fallback, especially on FD004
