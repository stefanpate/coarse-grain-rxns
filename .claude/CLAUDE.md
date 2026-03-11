# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cheminformatics + ML research project exploring ways to coarse-grain enzymatic reactions into SMARTS-encoded reaction rules. The pipeline: extract reaction templates from atom-mapped reactions → train a GNN to classify mechanism-relevant atoms → generate rules at varying levels of specificity for use in biosynthesis software (Pickaxe).

## Setup

```bash
uv sync
cp TEMPLATE_FILEPATHS.yaml configs/filepaths/filepaths.yaml
# Edit configs/filepaths/filepaths.yaml with actual data/artifact paths
```

## Commands

```bash
# Tests
pytest tests/
pytest tests/test_cheminfo.py::test_fingerprinter -v  # single test

# Scripts (Hydra-configured)
python scripts/write_rcr_rules.py R=1              # reaction center + R-hop rules
python scripts/write_rcr_rules.py -m R=1,2,3       # multiple R values
python scripts/write_mechinformed_rules.py          # M-CSA mechanism-informed rules
python scripts/hpo.py                              # hyperparameter optimization (Optuna)
python scripts/train_production.py                 # train production GNN model
python scripts/infer_mech_labels.py -m             # run inference
python scripts/write_mechinferred_rules.py          # rules from model predictions
python scripts/add_coreactant_roles.py mapped_rxns_fn=<file>.parquet
```

## Architecture

### Core Library (`src/cgr/`)

- **`rule_writing.py`** — Template extraction from atom-mapped reactions. Key fn: `extract_reaction_template()` builds SMARTS patterns with configurable atom neighborhood radius (R), stereo handling, and canonicalization.
- **`rxn_analysis.py`** — `ReactantGraph` (Pydantic model): represents reactants as node-feature matrix `V` + adjacency matrix `A`. Used as GNN input. Also contains `mcsplit()` for maximum common subgraph matching.
- **`ml.py`** — `GNN` (PyTorch Lightning): message-passing network that classifies which atoms belong in a reaction template. Wraps chemprop. `SklearnGNN` provides sklearn-compatible interface for calibration.
- **`featurize.py`** — Atom featurization schemes (Daylight atomic invariants, Morgan fingerprints with custom invariants) used to build `ReactantGraph.V`.
- **`cheminfo.py`** — Legacy utilities (marked for cleanup): fingerprinting, subgraph extraction, similarity.

### Data Flow

1. Atom-mapped reactions (Rhea/M-CSA) → `extract_reaction_template()` → SMARTS rules
2. Mechanistically-informed labels (M-CSA) → train GNN on atom classification
3. GNN inference on Rhea reactions → probability scores per atom
4. Scores + decision thresholds → `write_mechinferred_rules.py` → SMARTS rules

### Configuration

Hydra manages all script configs under `configs/`. Key configs:
- `configs/filepaths/filepaths.yaml` — all data/artifact paths (gitignored, copy from `TEMPLATE_FILEPATHS.yaml`)
- `configs/train_production.yaml`, `configs/hpo.yaml` — model training
- `configs/write_rcr_rules.yaml`, `configs/write_mechinferred_rules.yaml` — rule generation

The "distilled" rules come from M-CSA mechanism labels; "learned" rules come from GNN predictions on Rhea.

### Artifacts

- `artifacts/mlruns/` — MLflow experiment tracking
- `artifacts/hpo_studies/` — Optuna studies
- `artifacts/rules/` — generated SMARTS rule sets
