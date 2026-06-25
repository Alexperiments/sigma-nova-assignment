# Design doc

## Assumptions

- Since the assignment is asking for a benchmarking service, I'll assume the benchmark lists (currently only BNCI2014_001) to be extensible in the future. This implies abstraction over benchmarks.
- Since the training of a linear layer could be an expensive task, I'll assume the trained weights to be cached for future use. This could be locally or not.
- since LaBraM model comes in different size I'll assume that the base model is enough for the purpose of this assignment. In addition it has a similar size with respect to CBraMod.

## To be tested

- It's possible that running the models on the benchmark could be an expensive task, if this is the case it could be useful to cache the results for a given unique combination of (service version, benchmark, model, trained head).

## Requirements

- The service should be flexible regarding channel counts, sampling rates, window sizes, as there could be mismatches between models and datasets.