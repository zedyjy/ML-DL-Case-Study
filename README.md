# ML-DL-Case-Study

## LSTM pipeline (`LSTM/`)

Cross-age HAR: train an LSTM on younger adults (HARTH + RealWorld HAR),
evaluate zero-shot on older adults (HAR70+). Steps 1–6 in `LSTM/src/`
(plain `.py` with `# %%` cells): load each dataset, map to six common
labels, cut 128-sample windows with subject-grouped 5-fold CV, train the LSTM.

Full write-up, decisions and results: [`LSTM/docs/PIPELINE_AND_RESULTS.md`](LSTM/docs/PIPELINE_AND_RESULTS.md).

```bash
pip install -r requirements.txt
cd LSTM
python src/01_load_harth.py   # then 02 ... 06 in order
```

Raw data is read from the repo root (`harth/`, `har70plus/`, `Real world HAR/`);
`Real world HAR/` must be downloaded separately. Outputs go to `LSTM/processed/`
(git-ignored).
