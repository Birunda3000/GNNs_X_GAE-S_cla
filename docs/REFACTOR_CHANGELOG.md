# REFACTOR_CHANGELOG — Composition over Inheritance (Trainer Pattern)

**Date:** 2026-09-03
**Scope:** `src/models/**`, `src/experiment_runner.py`, `src/grid_search/**`, `src/early_stopper.py`, `src/config.py` (untouched)
**Result:** models became pure `nn.Module`s; data loading and training loops were extracted into `GraphDataModule` and a dependency-injected `Trainer`.

---

## 1. Motivation

The previous architecture was a deep inheritance hierarchy. `PyTorchClassifier` (and `BaseGAECommon` on the embedding side) inherited **both** the tensor math and the entire training pipeline:

- epoch loops, optimizer/scheduler creation, JIT compilation, early stopping, metric computation, and `CuGraphNeighborLoader` construction all lived *inside* the model classes;
- every concrete model (`GCNClassifier`, `GATClassifier`, `DynamicGNNClassifier`, `GCNGAE`, `GraphSageVGAE`, …) silently inherited hundreds of lines of training machinery it never overrode;
- hyperparameters were read implicitly via `self.config`, so a model could not be trained outside the project's `Config` object;
- two near-duplicate training loops existed (`PyTorchClassifier.internal_train_model` and `DynamicEmbeddingGNNClassifier.internal_train_model`), and a third on the GAE side — any change to stopping logic or sampling had to be replicated in three places.

This hurt **research reproducibility**: training configuration was entangled with model definition, making it impossible to swap an optimizer, a sampler, or a stopping criterion without editing model code, and making unit testing of the math nearly impossible.

The refactoring applies the **composition / Trainer pattern**:

- **Models** answer only *"how do I propagate tensors?"* (`forward` / `encode`).
- **`GraphDataModule`** answers only *"where do data, splits, and loaders come from?"*
- **`Trainer`** answers only *"how is a training run executed?"* — and receives every dependency (model, data module, optimizer, criterion, scheduler, device) via its constructor.

---

## 2. Structural changes

### 2.1 Models are now "dumb" `nn.Module`s

| File | What changed |
|---|---|
| `src/models/pytorch_classification/base_classifiers.py` | `PyTorchClassifier` reduced to `__init__` (device/name) + `forward` stub + a minimal `inference()`. **Removed:** `train_model`, `internal_train_model`, `_train_step`, `evaluate`, `verify_train_input_data`, loader construction, JIT calls, sklearn/tqdm imports. |
| `src/models/pytorch_classification/classification_models.py` | Removed `verify_train_input_data` overrides and dead imports. `MLPClassifier`, `GCNClassifier`, `GATClassifier` now contain only layers + `forward`. |
| `src/models/pytorch_classification/dynamic_gnn.py` | Removed `DynamicGNNClassifier.verify_train_input_data` and the full-batch `internal_train_model` of `DynamicEmbeddingGNNClassifier` (and its duplicate import block). |
| `src/models/embedding_models/base_graph_autoenconders_model.py` | `BaseGAECommon` is now a pure `nn.Module`. **Removed:** `train_model`, `verify_train_input_data`, `evaluate`, `BaseModel` inheritance. **Kept:** `decode`, `reconstruction_loss`, `encode`/`compute_total_loss` (stubs), `inference`. |
| `src/models/base_model.py` | `BaseModel` no longer enforces `train_model`/`evaluate`/`inference` via `@abstractmethod` (only `SklearnClassifier` still inherits it). JIT helpers (`compile_methods`/`decompile_methods`) were removed — the `Trainer` now owns compilation. |

**Kept on the models** (deliberately): `model_name`, `device`, the `use_gnn` class flag, `num_layers`, and `inference()` — these are queried by the runner, the early stopper, and the data module wiring, and removing them would break callers without buying separation.

### 2.2 `src/datamodule.py` — new

`GraphDataModule` owns everything data-related:

- wraps a finished `torch_geometric.data.Data` (splits are already attached by `src/data_converters.py`);
- `prepare(device)` — validates required tensors (`y`, `train/val/test_mask`, features) and moves the full graph to the device;
- `train_dataloader()` — yields a `CuGraphNeighborLoader` (NGC `cugraph` backend, `NeighborLoader` fallback), with `num_neighbors = [k] * num_layers` (`num_layers=0` → neighbor-free loader for the MLP path);
- `full_batch()` — the full graph for evaluation.

### 2.3 `src/trainer.py` — new

`Trainer` receives **everything** through its constructor:

```python
Trainer(model, datamodule, optimizer, criterion, scheduler, device)
```

and exposes two entry points:

- `fit(epochs, early_stopper)` — supervised training. Dispatches to a mini-batch loop (`CuGraphNeighborLoader`, for `GCN`/`GAT`/`DynamicGNN`/`MLP`) or a full-batch loop (for `EmbeddingBag` GNNs such as `FacebookEmbeddingGNN`), based on the data signature. Per-epoch: train step → full-batch `evaluate` on train/val masks → `early_stopper.check(...)` (post-epoch hook) → `scheduler.step(...)`. Returns the exact dictionary shape consumed by `ExperimentRunner`.
- `fit_gae(epochs, early_stopper, scheduler_metric_name=None)` — unsupervised link-reconstruction training (GAE/VGAE), faithful port of the removed `BaseGAECommon.train_model` (autocast/FP8 support, grad clipping, `DeviceTimer`). Returns `best_epoch`, `best_score`, `best_scores`, `training_history`, `total_training_time`.

The JIT compile/decompile of `forward`/`encode`/`decode` now happens inside the `Trainer` (with `finally`-guaranteed cleanup), never on the model.

### 2.4 Wiring

- `src/experiment_runner.py` — the PyTorch branch no longer calls `model.train_model(data)`; it builds `GraphDataModule` + `Adam` + `CrossEntropyLoss` + `ReduceLROnPlateau` + `EarlyStopper`, instantiates the `Trainer`, and calls `trainer.fit()` inside the existing `memory_usage` profiler. The sklearn branch is untouched.
- `src/grid_search/gnn_classifiers_optuna.py` — Optuna objective now trains through `Trainer.fit()`.
- `src/grid_search/gae_optuna_grid.py` — Optuna objective now trains through `Trainer.fit_gae()` with a `UniversalEarlyStopper` wrapping a `LogRegMetric` probe.

### 2.5 Bug fixes surfaced by the refactoring

1. **Dangling `EarlyStopper` import** — `src/early_stopper.py` had been migrated to `UniversalEarlyStopper` in an earlier commit, but the classification and grid-search paths still imported the deleted `EarlyStopper` class. The classic interface was re-added alongside `UniversalEarlyStopper` (Step 3).
2. **`evaluate_embeddings` did not exist** — `gae_optuna_grid.py` referenced an undefined function as its stopping criterion; replaced with the defined `LogRegMetric` probe.
3. **`best_score` key mismatch** — the GAE grid search read `training_report["best_score"]`, which the GAE loop never returned; `Trainer.fit_gae()` now emits it.
4. **EmbeddingBag inference crash** — `inference()` forwarded `(data.x, data.edge_index)`, but sparse-feature data has no `x`; it now branches on `feature_indices`.

---

## 3. How to run a training run now

Three steps: instantiate the module, assemble its dependencies, call `.fit()`.

```python
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.config import Config
from src.datamodule import GraphDataModule
from src.trainer import Trainer
from src.early_stopper import EarlyStopper
from src.models.pytorch_classification.dynamic_gnn import DynamicGNNClassifier

config = Config()

# 1. Pure model (tensor propagation only)
model = DynamicGNNClassifier(
    config=config, input_dim=128, output_dim=7,
    layer_type=..., num_layers=3, hidden_dim=256,
).to(config.DEVICE)

# 2. Data + training dependencies (dependency injection)
datamodule = GraphDataModule(data, num_layers=model.num_layers, batch_size=1024)
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=config.SCHEDULER_PATIENCE)
early_stopper = EarlyStopper(patience=config.EARLY_STOPPING_PATIENCE,
                             min_delta=config.EARLY_STOPPING_MIN_DELTA,
                             mode="max", metric_name="val_f1")
trainer = Trainer(model, datamodule, optimizer,
                  criterion=nn.CrossEntropyLoss(),
                  scheduler=scheduler, device=config.DEVICE)

# 3. Train
report = trainer.fit(epochs=config.EPOCHS, early_stopper=early_stopper)
```

For an unsupervised GAE/VGAE run, step 3 becomes:

```python
from src.early_stopper import UniversalEarlyStopper
from src.embeddings_eval import LogRegMetric

stopper = UniversalEarlyStopper(metrics=[LogRegMetric(patience=32)],
                                stop_condition="all", restore_best=True)
report = trainer.fit_gae(epochs=config.EPOCHS, early_stopper=stopper,
                         scheduler_metric_name="LogReg")
```

`report` keeps the keys consumed downstream (`test_accuracy`, `test_f1`, `*_report`, `*_confusion_matrix`, `best_epoch`, `training_history`, `total_training_time`), so `ExperimentRunner`, `ReportManager`, and the Optuna objectives required no contract changes.

---

## 4. Reproducibility notes

- **Fixed hyperparameter surface.** Every training input (epochs, learning rate, patience, batch size, sampling fan-out, device) is now an explicit argument at the call site — no implicit reads of `Config` inside the model.
- **Testability.** The tensor math can be unit-tested without GPU, loaders, or stopping logic; the loop can be tested against a mock model.
- **Behavior parity.** The two smoke tests executed during this refactoring confirm the trained-report dictionary shape and the early-stopping metrics are unchanged from the legacy loops.
