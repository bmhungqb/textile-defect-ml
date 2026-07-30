# textile-defect-ml

RF-DETR object detection for textile defects, organized around **dataset versioning**:
one dataset version → many experiments → one released best model.

```
Label Studio + GCS ──▶ output/<version>/            ──▶ experiments ──▶ release
   run_data_pipeline      info.json, dataset/          run_training_    run_release_
                                                         pipeline         pipeline
```

Everything belonging to a dataset version lives in one folder:

```
output/v20260729_145230/
├── info.json                             # version, provenance, class distribution per split
├── label_studio_tasks.json               # the task export, exactly as downloaded
├── raw_annotations.json                  # that export converted to COCO
├── dataset/
│   ├── train/ valid/ test/               # images + _annotations.coco.json (RF-DETR layout)
│   └── train.json val.json test.json
├── experiment_20260729_150000_trial_0/   # Optuna search: one folder per trial
│   …                                     # params.json, metrics.csv, checkpoints
├── experiment_20260729_150000_trial_19/
├── experiment_20260730_090000/           # single fixed-config run
├── experiments.md / experiments.csv      # comparison table (written at release time)
└── release.json / RELEASE_NOTES.md
```

---

## 1. Setup (once)

Requires Python 3.11–3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                    # creates .venv and installs everything
```

Create `.env` from the template and fill it in:

```bash
cp .env.example .env
```

| variable                         | what it's for                                                                       |
| -------------------------------- | ----------------------------------------------------------------------------------- |
| `LABEL_STUDIO_URL`               | Label Studio instance, e.g. `https://labelstudio.laka.ai`                            |
| `LABEL_STUDIO_API_KEY`           | your Label Studio account token                                                      |
| `GOOGLE_APPLICATION_CREDENTIALS` | path to the GCP service-account JSON key, used to download images and upload models  |

All three pipelines load `.env` automatically — you don't need to export anything.

**GCS credentials.** A relative path (`vv-credentials.json`) is resolved against the
project root, so it works no matter which directory you launch from. On a server, put
the key wherever you keep secrets and give the absolute path:

```bash
GOOGLE_APPLICATION_CREDENTIALS=/etc/secrets/vv-credentials.json
```

If the variable is unset, the client falls back to Application Default Credentials — the
attached service account on a GCE/GKE instance. If it's set but the file is missing, the
run fails immediately with the path it tried, rather than falling back and failing later
with a 403.

Two optional extras:

- **Weights & Biases** — training logs to wandb by default. Run `wandb login` once, or
  set `WANDB_MODE=offline` in `.env` to skip it.
- **GitHub CLI** — only needed for the last step: `brew install gh && gh auth login`.
  Without it, use `--no-github` and create the release by hand.

Pretrained weights: [configs/training.yaml](configs/training.yaml) points `pretrain_weights` at
`weights/auto_training_data4training_distill_weights_last.ckpt`. Put that file there, or
blank the key out to start from RF-DETR's own published weights.

---

## 2. Build a dataset version

Pulls reviewed tasks from Label Studio, splits them by each image's own `split` field,
downloads the images from GCS, and writes `info.json`.

```bash
uv run python pipelines/run_data_pipeline.py --version v1
```

| flag        | meaning                                                                                             |
| ----------- | --------------------------------------------------------------------------------------------------- |
| `--version` | version label. Omit it to use `version:` from the config, or leave that blank for `v<timestamp>`     |
| `--config`  | default `configs/data.yaml`                                                                          |

Configure the source in [configs/data.yaml](configs/data.yaml): `label_studio.project_id`,
the `defect_classes` map, and `output_dir` (the root holding all versions).

**Output:** `output/v1/`, and a log line with the totals:

```
Dataset version v1: 1050 images, 2749 annotations, distribution {'cham_do': 488, ...}
```

> ⚠️ Reusing a version label writes into the same folder. Bump `--version` (or blank out
> `version:` in the config so each pull gets its own timestamp) when you pull fresh labels.

---

## 3. Train experiments on that version

Each run writes `experiment_<datetime>/` **inside** the dataset version folder, so
experiments never lose track of the data they were trained on. Run this as many times as
you like per version.

**Single run** with the fixed hyperparameters in `fixed_params`:

```bash
uv run python pipelines/run_training_pipeline.py output/v1
```

**Optuna search** (multi-objective over mAP50 / recall / F1, `optuna.n_trials` trials):

```bash
uv run python pipelines/run_training_pipeline.py output/v1 --optuna
```

| flag       | meaning                                                                     |
| ---------- | --------------------------------------------------------------------------- |
| `--optuna` | run a search instead of a single run (overrides `use_optuna` in the config)  |
| `--config` | default `configs/training.yaml`                                              |

Tunables in [configs/training.yaml](configs/training.yaml):

| key                                       | meaning                                                                                                        |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `project_name`                            | wandb project. Runs are named `<data_version>/experiment_<datetime>[/trial_<n>]`                                |
| `weighted_dataloader`                     | `true` → inverse-class-frequency sampling on train (see `src/training/dataloader.py`)                           |
| `pretrain_weights`                        | checkpoint to initialize from; blank → RF-DETR defaults                                                         |
| `fixed_params`                            | hyperparameters for a single run — every key must be a `rfdetr.config.TrainConfig` field                        |
| `optuna.n_trials`, `optuna.search_space`  | the search                                                                                                      |
| `optuna.fixed_params`                     | applied to every trial — everything the search doesn't tune. If a key is in both blocks, the tuned value wins   |

Early stopping is on for both paths (`early_stopping: true`, patience 10, min delta
0.001, tracking the EMA metric `val/ema_mAP_50_95`). Lower `early_stopping_patience`
under `optuna.fixed_params` to shorten a search.

> ⚠️ Write exponents with the decimal point (`1.0e-4`, **not** `1e-4`). PyYAML parses the
> bare form as a *string*, which silently breaks the optimizer.

Every run — a single run or one Optuna trial — gets its own top-level folder in the
dataset version: `experiment_<datetime>/` or `experiment_<datetime>_trial_<n>/`.

**Output per run:** `checkpoint_best_total.pth` (best of regular vs EMA, optimizer state
stripped — this is the one you serve), `checkpoint_best_ema.pth`,
`checkpoint_best_regular.pth`, `checkpoint.pth` (last epoch, large), `metrics.csv`,
`params.json`. Each run's results are read back from its own `metrics.csv`, so a search
that dies partway still leaves every finished trial releasable.

**In wandb:** each run logs separately, named after its folder
(`experiment_<datetime>` or `experiment_<datetime>_trial_<n>`).

---

## 4. Release the version's best model

Compares **every run across every experiment** of one dataset version, picks the best,
pushes it to GCS, and drafts a GitHub release.

Set your bucket in [configs/release.yaml](configs/release.yaml) first:

```yaml
gcs:
  bucket: "textile-datasets"
  prefix: "textile-defect-models/VV"
select_by: F1        # or mAP50 | ema_mAP50 | recall
```

Preview without touching anything — prints the table and the exact release notes:

```bash
uv run python pipelines/run_release_pipeline.py output/v1 -m "Baseline after re-labelling nep_nhan." --dry-run
```

Then release for real:

```bash
uv run python pipelines/run_release_pipeline.py output/v1 -m "Baseline after re-labelling nep_nhan."
```

| flag              | meaning                                                                            |
| ----------------- | ------------------------------------------------------------------------------------ |
| `-m`, `--message` | your note, shown at the top of the release                                          |
| `--dry-run`       | build the table and notes, upload nothing                                           |
| `--version`       | release version; defaults to the dataset version                                    |
| `--index N`       | release row `N` of the table instead of the best-scoring run                        |
| `--select-by`     | metric to rank by, overriding the config                                            |
| `--bucket`        | GCS bucket, overriding the config                                                   |
| `--full`          | upload the whole run folder, including the large `checkpoint.pth`                   |
| `--publish`       | publish the release instead of leaving it a draft                                   |
| `--no-github`     | upload to GCS only, and write `RELEASE_NOTES.md` to the version folder              |

**Selection rule:** highest `select_by` metric among runs that actually produced a
checkpoint — a crashed trial with a great score can't be released. The metric for each
run is read from its `metrics.csv` at the epoch with the best `val/F1`.

**What lands in GCS** (`gs://<bucket>/<prefix>/<version>/`): the winning run's
checkpoints, `metrics.csv`, `params.json`, plus the dataset version's `info.json` and
`label_studio_tasks.json`, and the `experiments.csv`, `experiments.md` and
`release.json` for the release. A version folder with no `label_studio_tasks.json`
(built before it was kept) logs a warning and publishes the rest.

**What lands on GitHub:** a draft release tagged `model-<version>` containing only
Markdown — your message, the headline metrics, the dataset distribution table, the
`gs://` download links, and the full experiment comparison. No files are attached; the
artifacts stay in GCS. The target repo is read from your `origin` git remote unless
`github.repo` is set in the config.

Releasing the same dataset version twice needs a distinct version, e.g.
`--version v1-r2` (otherwise `gh` refuses the duplicate tag).

---

## Full example

```bash
uv sync && cp .env.example .env         # fill in .env

uv run python pipelines/run_data_pipeline.py --version v1
uv run python pipelines/run_training_pipeline.py output/v1 --optuna
uv run python pipelines/run_training_pipeline.py output/v1     # another experiment, same data
uv run python pipelines/run_release_pipeline.py output/v1 -m "First release" --dry-run
uv run python pipelines/run_release_pipeline.py output/v1 -m "First release"
```

The release body's comparison table looks like this, with the released run in **bold**:

| index | experiment                     | trial | lr         | lr_scheduler | mAP50    | recall   | F1        |
| ----- | ------------------------------ | ----- | ---------- | ------------ | -------- | -------- | --------- |
| 0     | experiment_20260729_150000     | 0     | 0.00042    | cosine       | 0.58     | 0.60     | 0.585     |
| 1     | experiment_20260729_150000     | 1     | 3.1e-05    | step         | 0.63     | 0.65     | 0.635     |
| **2** | **experiment_20260730_090000** | **—** | **0.0001** | **cosine**   | **0.66** | **0.68** | **0.665** |

Only hyperparameters that actually differ between runs get a column; `—` in `trial`
means a single fixed-config experiment rather than an Optuna trial.

---

## Layout

| path                                 | what it is                                                        |
| ------------------------------------ | ------------------------------------------------------------------ |
| `pipelines/run_data_pipeline.py`     | Label Studio + GCS → `output/<version>/`                          |
| `pipelines/run_training_pipeline.py` | one experiment against a dataset version                          |
| `pipelines/run_release_pipeline.py`  | compare experiments → GCS → GitHub release                        |
| `src/data/build_dataset.py`          | COCO conversion, splitting, image download, `info.json`           |
| `src/training/train.py`              | RF-DETR training, single run and Optuna search                    |
| `src/training/dataloader.py`         | class-imbalance-aware sampler                                     |
| `src/release/summarize.py`           | experiment table, best-run selection                              |
| `src/release/publish.py`             | GCS upload, release notes, `gh release create`                    |
| `src/utils/`                         | GCS, Label Studio, logging helpers                                |

Note: `model: RFDETRMedium` in `configs/training.yaml` is informational —
`src/training/train.py` instantiates `RFDETRMedium` directly. Change the class there to
switch model size.

---

## Troubleshooting

| symptom                                                    | fix                                                                                        |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `No info.json in …`                                        | you passed a plain folder; pass a dataset version folder built by the data pipeline         |
| `No dataset/ folder in …`                                  | the data pipeline didn't finish — rerun it                                                  |
| `No experiment_* folders under …`                          | nothing has trained against this version yet                                                |
| `No run … has both a checkpoint and a F1 score`            | every run crashed before saving; check `metrics.csv` in the experiment folders              |
| `GCS credentials file not found: …`                        | fix `GOOGLE_APPLICATION_CREDENTIALS` in `.env` — use an absolute path on a server            |
| `403` / `Anonymous caller` from GCS                        | the key file is valid but its service account lacks access to the bucket                    |
| `No GCS bucket configured`                                 | set `gcs.bucket` in `configs/release.yaml` or pass `--bucket`                                |
| `gh CLI not found`                                         | `brew install gh && gh auth login`, or use `--no-github`                                    |
| optimizer behaving strangely                               | check for bare `1e-4` exponents in the YAML config                                           |
