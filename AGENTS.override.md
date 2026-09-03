Here is the translation of your guidelines into English:

# Refactoring Guidelines (Graph Machine Learning)

* **Architectural Objective:** Strictly migrate to the Composition over Inheritance principle. Separate the neural network definition from the training loop.
* **Performance (Immutable):** Preserve all existing performance optimizations. Do not remove the TorchDynamo JIT compilation decorators, Transformer Engine (FP8) support, and NVIDIA RAPIDS imports.
* **Typing:** Maintain strict static typing (Type Hints) across all method signatures.
* **Golden Rule:** Do not alter the mathematical logic of the models (loss functions, forward passes). Focus solely on the orchestration and OO structure.
* **Ignore "data" and "old" folders:** Ignore "data" and "old" folders.