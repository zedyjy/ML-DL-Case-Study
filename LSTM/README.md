# LSTM

2-layer LSTM (64 hidden units) on raw accelerometer windows, PyTorch.

- Input: `preprocessing/processed/windows.npz` and `scalers.json`. Run
  `preprocessing/src/01` to `05` first.
- Settings for this model only: `lstm_config.py` (model size, epochs,
  learning rate, rotation augmentation, which datasets to train on).
- Run: `python LSTM/06_train_lstm.py` (~35 min on an Apple GPU).
- Output: `LSTM/results/` (metrics, confusion matrices) and `LSTM/models/`
  (weights), both git-ignored.

Model details and results: section 5 of
[`preprocessing/docs/PIPELINE_AND_RESULTS.md`](../preprocessing/docs/PIPELINE_AND_RESULTS.md).
