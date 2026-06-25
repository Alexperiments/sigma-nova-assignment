# Design doc

## Assumptions

- Since the assignment is asking for a benchmarking service, I'll assume the benchmark lists (currently only BNCI2014_001) to be extensible in the future. This implies abstraction over benchmarks.
- Since the training of a linear layer could be an expensive task, I'll assume the trained weights to be cached for future use. This could be locally or not.
- since LaBraM model comes in different size I'll assume that the base model is enough for the purpose of this assignment. In addition it has a similar size with respect to CBraMod
- since the assignment talks only about foundation model benchmarking, so all the future models are also assumed to be foundation models. A new linear layer will always be trained on top of the FM.
- since the provided models' checkpoints target incompatible Braindecode versions, I'll assume that the latest version has to be used. So I won't be able to use directly the conversion code showed [here](https://huggingface.co/braindecode/Labram-Braindecode), but I'll use the [version](https://huggingface.co/braindecode/labram-pretrained) provided by the latest Braindecode package.

## To be tested

- It's possible that running the models on the benchmark could be an expensive task, if this is the case it could be useful to cache the results for a given unique combination of (service version, benchmark, model, trained head).

## Requirements

- The service should be flexible regarding channel counts, sampling rates, window sizes, as there could be mismatches between models and datasets.

## Design decisions

- The application will be a CLI, for simplicity given the time constrain, portability and versatility.
- The output for each benchmark run will be a markdown file containing the run configs, metadata and results. In addition it will output a result file containing only the numerical data in tabular format.
- The benchmark layer will follow an object-oriented API with a shared base class and concrete dataset implementations. For the first version, evaluation will focus on aggregate metrics only (accuracy and Cohen's kappa), without subject-wise or session-wise split reporting.

## Project structure

src/
└── eeg_benchmark/
    ├── __init__.py
    ├── cli.py
    ├── runner.py
    ├── probe.py
    ├── metrics.py
    ├── results.py
    │
    ├── datasets/
    │   ├── __init__.py
    │   ├── base.py
    │   └── bnci2014_001.py
    │   └── ...
    │
    └── models/
        ├── __init__.py
        ├── base.py
        ├── labram.py
        └── cbramod.py
        └── ...

## Future improvements

- Add support for subject-wise and session-wise evaluation splits.
- Introduce configurable preprocessing and loading options through the dataset config.
- Improve reporting with per-subject breakdowns, confidence intervals and richer experiment metadata.