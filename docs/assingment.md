# ML Capability Interview

## EEG Benchmarking Service — Take-Home Exercise

## Context

At Sigma Nova, we regularly evaluate EEG foundation models against each other on BCI tasks. We want a service that makes this repeatable and accessible to the team.

Your task is to build a benchmarking service for EEG motor imagery classification using the models and dataset linked below. How you design and expose the service is up to you.

This exercise is meant to take 1–2 days of work. You may simplify parts or focus on others; prioritization is part of the skills we are evaluating.

## Data & Models

**Dataset:** Use the [BCI Competition IV Dataset 2a](https://www.bbci.de/competition/iv/desc_2a.pdf) (`BNCI2014_001`), a widely used motor imagery benchmark with 22-channel EEG recordings from 9 subjects.

It is publicly available and can be downloaded automatically via [braindecode](https://braindecode.org/stable/index.html)'s `MOABBDataset` class.

If compute is limiting, you may limit yourself to a subset of subjects.

The task is **4-class classification**: left hand, right hand, feet, and tongue motor imagery.

**Models:** Your API should support following models, based on `braindecode`:

- [LaBraM](https://huggingface.co/braindecode/Labram-Braindecode)
- [CBraMod](https://huggingface.co/braindecode/cbramod-pretrained)

Each model's backbone should be used **frozen**. You should train a linear classification head on top of extracted embeddings (**linear probing**) using training split.

## Note

Foundation models are typically pretrained on specific EEG configurations (channel counts, sampling rates, window sizes) that may differ from this dataset. How you handle this mismatch should be discussed in your report.

You are not expected to perform extensive preprocessing or parameter tuning beyond what is necessary to run models.

## Deliverables

A Git repository with your code, a `README` explaining how to run it, and a document detailing main decisions you made and what you would improve given more time.

The service should be directly runnable and installable.

## Evaluation Criteria

Your solution will be evaluated on following criteria:

- Solution architecture
- Code quality and maintainability
- Robustness
- Efficiency
- Reproducibility
- Clarity of report

## Closing Notes

Good luck! We're looking forward to seeing your solution, and discussing it with you.

This is first edition of this test. Let us know if you encounter any blocking issues, if anything is unclear, or if test does not seem completeable within reasonable timeframe.
