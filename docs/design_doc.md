# Design doc

## Assumptions

- Since the assignment is asking for a benchmarking service, I'll assume the dataset list (initially BNCI2014_001) to be extensible in the future. This implies abstraction over datasets.
- since LaBraM model comes in different size I'll assume that the base model is enough for the purpose of this assignment. In addition it has a similar size with respect to CBraMod
- since the assignment talks only about foundation model benchmarking, so all the future models are also assumed to be foundation models. A new linear layer will always be trained on top of the FM.
- since the provided models' checkpoints target incompatible Braindecode versions, I'll assume that the latest version has to be used. So I won't be able to use directly the conversion code showed [here](https://huggingface.co/braindecode/Labram-Braindecode), but I'll use the [version](https://huggingface.co/braindecode/labram-pretrained) provided by the latest Braindecode package.
- I'll assume here braindecode datasets only.

## Requirements

- The service should run a motor imagery benchmark on BNCI2014_001.
- The service should support frozen LaBraM and CBraMod backbones.
- The service should train a linear classification head on top of extracted embeddings using the training split.
- The project should be directly runnable and installable.

## Design decisions

- The application will be a CLI, for simplicity given the time constraint, portability and versatility.
- The output for each benchmark run will be a csv file containing only the numerical data in tabular format, ready to be read with pandas for example.
- The project will keep the dataset abstraction small: a MOABB dataset wrapper will resolve the benchmark name, load the Braindecode/MOABB dataset, expose metadata and create event-based windows.
- The model abstraction will be object-oriented: a shared frozen foundation-model base class will handle device placement/freezing, while concrete model adapters will implement the encoding details.
- For the first version, evaluation will focus on aggregate accuracy only. Train/test data will be split by the dataset session labels, but reporting will not include per-subject or per-session metric breakdowns.
- The assignment asks the report to discuss possible mismatch between a foundation model's pretraining EEG configuration and the benchmark dataset. For this version, the implementation will handle the concrete mismatch needed to run the selected models: event windows will be adapted to the model input length, and LaBraM will receive the dataset channel names so Braindecode can select the matching positional embeddings.
- Window-size adaptation will be deterministic: windows longer than the model input length will be cropped, while windows shorter than the model input length will be padded with zeroes.
- Each result row will include the benchmark configuration needed to interpret the run: model, dataset, seed, device, epochs, learning rate, batch size, embedding shape and target window size.
- Dataset metadata such as channel names and event labels will be inferred from the loaded data when possible.
- The dataset layer will be responsible for loading the dataset, applying minimal generic preprocessing, exposing metadata and creating windows from events.
- The model-compatibility layer will be responsible for adapting event windows to model requirements, while the model layer remains focused on encoding and inference.
- Minimal preprocessing will pick EEG channels, convert signal units from volts to microvolts and clip obviously extreme amplitudes. This should keep the benchmark inputs in a reasonable range without introducing dataset-specific feature engineering.
- For CBraMod, the channel and patch feature map will be mean-pooled into one embedding per trial to keep this assignment implementation simple and fast to compute.

## Future improvements

- Add subject-wise metric reporting.
- Add more dataset-specific loading/preprocessing overrides only when automatic metadata inference is insufficient.
- Add confidence intervals, for example seeding different runs and computing average and std. dev.
- The current benchmark accuracy is low and close to chance for a 4-class task. This may be due to a combination of minimal preprocessing, simple crop/pad window handling, and limited linear-probe tuning. A useful next step would be to run targeted ablations around preprocessing, window extraction, and probe hyperparameters to understand which part is limiting performance.
- The first version will support MOABB datasets through Braindecode only, and could later be extended to other datasets served by Braindecode.
- The persistence for now is just a simple CSV file for run, a much more robust and maintanable solution would be to store the benchmark runs in a relational database.
- The linear probes are trained for the same fixed amount of epochs for each model, a smarter approach would be to implement an early stopping to prevent overfitting.
