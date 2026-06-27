# Design doc

## Assumptions

- Since the assignment is asking for a benchmarking service, I'll assume the benchmark list (initially BNCI2014_001) to be extensible in the future. This implies abstraction over benchmarks.
- since LaBraM model comes in different size I'll assume that the base model is enough for the purpose of this assignment. In addition it has a similar size with respect to CBraMod
- since the assignment talks only about foundation model benchmarking, so all the future models are also assumed to be foundation models. A new linear layer will always be trained on top of the FM.
- since the provided models' checkpoints target incompatible Braindecode versions, I'll assume that the latest version has to be used. So I won't be able to use directly the conversion code showed [here](https://huggingface.co/braindecode/Labram-Braindecode), but I'll use the [version](https://huggingface.co/braindecode/labram-pretrained) provided by the latest Braindecode package.
- I'll assume here braindecode datasets only.

## To be tested

- For this first exercise, I'll assume the benchmark can be recomputed directly, including model embeddings, probe training and result generation.


## Requirements

- The service should be explicit about possible mismatches between models and datasets: channel count, sampling frequency and window size.
- For this exercise, I won't add preprocessing dedicated to channel-count or sampling-frequency mismatches. If channel-count mismatches need to be supported later, I would rely on the interpolated versions of the Braindecode models where available. If sampling-frequency mismatches need to be supported later, I would resample the raw signal before creating windows.
- Window-size mismatches will be handled conservatively by adapting the trial windows to the model input length.

## Design decisions

- The application will be a CLI, for simplicity given the time constraint, portability and versatility.
- The output for each benchmark run will be a csv file containing only the numerical data in tabular format, ready to be read with pandas for example.
- The project will keep the dataset abstraction small: a MOABB dataset wrapper will resolve the benchmark name, load the Braindecode/MOABB dataset, expose metadata and create event-based windows.
- The model abstraction will be object-oriented: a shared frozen foundation-model base class will handle device placement/freezing, while concrete model adapters will implement the encoding details.
- For the first version, evaluation will focus on aggregate accuracy only. Train/test data will be split by the dataset session labels, but reporting will not include per-subject or per-session metric breakdowns.
- Window-size adaptation will be deterministic: windows longer than the model input length will be cropped, while windows shorter than the model input length will be padded with zeroes.
- Each result row will include the benchmark configuration needed to interpret the run: model, dataset, seed, device, epochs, learning rate, batch size, embedding shape and target window size.
- Dataset metadata such as channel names, sampling rate and effective window size will be inferred from the loaded data when possible. Explicit constructor arguments are only fallback/override hooks, not a required configuration object.
- The dataset layer will be responsible for loading the dataset, applying minimal generic preprocessing, exposing metadata and creating windows from events.
- The model-compatibility layer will be responsible for adapting event windows to model requirements, while the model layer remains focused on encoding and inference.
- Minimal preprocessing will pick EEG channels, convert signal units from volts to microvolts and clip obviously extreme amplitudes. This should keep the benchmark inputs in a reasonable range without introducing dataset-specific feature engineering.


## Project structure

src/
└── eeg_benchmark/
    ├── __init__.py
    ├── adapters.py
    ├── cli.py
    ├── probe.py
    ├── pipeline.py
    ├── results.py
    ├── datasets.py
    ├── models.py
    └── types.py

## Future improvements

- Add subject-wise metric reporting.
- Add more dataset-specific loading/preprocessing overrides only when automatic metadata inference is insufficient.
- Improve reporting with per-subject breakdowns, confidence intervals and richer experiment metadata.
- The current benchmark accuracy is low and close to chance for a 4-class task. This may be due to a combination of minimal preprocessing, simple crop/pad window handling, CBraMod feature pooling, and limited linear-probe tuning. A useful next step would be to run targeted ablations around preprocessing, window extraction, embedding aggregation, and probe hyperparameters to understand which part is limiting performance.
- The first version will support MOABB datasets through Braindecode only, and could later be extended to other datasets served by Braindecode.
- The persistence for now is just a simple CSV file for run, a much more robust and maintanable solution would be to store the benchmark runs in a relational database.
