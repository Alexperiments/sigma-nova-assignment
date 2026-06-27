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
