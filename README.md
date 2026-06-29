# sigma-nova-assignment

## Usage

Install dependencies:

```bash
uv sync
```

Run the default benchmark:

```bash
uv run eeg-benchmark
```

Run with explicit options:

```bash
uv run eeg-benchmark --models labram cbramod --dataset BNCI2014_001 --device cpu --epochs 20 --lr 0.001
```

Results are written as CSV files under `results/`.

For reproducibility, model adapters pin specific Hugging Face revisions, and each result row records the Python, torch, Braindecode and MOABB versions used for the run.

## Extending

To add another MOABB dataset, add its MOABB dataset name to `DATASETS` in `src/eeg_benchmark/datasets.py`. If the dataset does not use the same session labels as `BNCI2014_001`, update `split_train_test` accordingly.

To add another Braindecode model, add a small adapter in `src/eeg_benchmark/models.py` that subclasses `FrozenFoundationModel`, loads the Braindecode model, implements `encode(...)` to return one embedding per trial, and registers it in `MODELS`. Pin the model repository revision when loading from Hugging Face.

## Tests

Run the unit tests:

```bash
uv run pytest
```
