# Feedback 

- The provided checkpoints target incompatible Braindecode versions: LaBraM loads directly with 1.3.x, while CBraMod requires 1.4+. It would help to specify a supported Braindecode version or provide mutually compatible checkpoints/loading instructions.
- In addition the provided Labram conversion code results in a model with a classification head already in place. I'd change the link to https://huggingface.co/braindecode/labram-pretrained. 
- If a specific version of the models is needed I'd make it clearer, either providing direct links or small scripts to reproduce the model's exact state. If any model going by those names is fine I'd also make it clearer.