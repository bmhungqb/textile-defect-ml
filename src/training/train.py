"""Train RF-DETR, with or without Optuna, with an optional weighted dataloader.
"""

from pathlib import Path

import optuna
import pandas as pd
from rfdetr import RFDETRMedium

from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_training(
    params: dict,
    dataset_dir: str,
    output_dir: str,
    weighted_dataloader: bool,
    project_name: str,
    run_name: str,
    pretrain_weights: str | None = None,
) -> str:
    """Run a single RF-DETR training with the given hyperparameters.

    Args:
        params: training hyperparameters, all of which must be TrainConfig fields
        dataset_dir: path to the dataset directory
        output_dir: path to the output directory
        weighted_dataloader: whether to use a weighted dataloader to handle class imbalance (requires training.dataloader.WeightedRFDETRDataModule)
        project_name: the name of the Weights & Biases project
        run_name: this run's wandb run name (e.g. "run_<timestamp>" or "run_<timestamp>/trial_<n>")
        pretrain_weights: optional checkpoint to initialize from (a model-level option)
    Returns:
        path to output_dir
    """
    model = RFDETRMedium(**({"pretrain_weights": pretrain_weights} if pretrain_weights else {}))

    train_kwargs = dict(
        params,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        log_per_class_metrics=True,
        wandb=True,
        tensorboard=True,
        project=project_name,
        run=run_name,
    )

    if weighted_dataloader:
        from src.training.dataloader import WeightedRFDETRDataModule
        from rfdetr import RFDETRModelModule, build_trainer
        from rfdetr.config import TrainConfig

        train_config = TrainConfig(**train_kwargs)
        module = RFDETRModelModule(model_config=model.model_config, train_config=train_config)
        datamodule = WeightedRFDETRDataModule(model_config=model.model_config, train_config=train_config)
        trainer = build_trainer(train_config, model.model_config)
        trainer.fit(module, datamodule)
    else:
        model.train(**train_kwargs)

    return str(Path(output_dir))


def _best_epoch_objectives(output_dir: str) -> tuple[float, float, float]:
    """Read metrics.csv to get mAP50/recall/F1 at the best epoch.
    Args:
        output_dir: path to the output directory where metrics.csv is saved
    Returns:
        tuple of (mAP50, recall, F1) at the best epoch, or (0.0, 0.0, 0.0) if metrics.csv is missing
    """
    metrics_path = Path(output_dir) / "metrics.csv"
    if not metrics_path.exists():
        return 0.0, 0.0, 0.0
    df = pd.read_csv(metrics_path)
    df = df[df["val/mAP_50"].notna()]
    if df.empty:
        return 0.0, 0.0, 0.0
    best_row = df.loc[df["val/F1"].idxmax()]
    return float(best_row["val/ema_mAP_50"]), float(best_row["val/recall"]), float(best_row["val/F1"])

def _suggest_params(trial: optuna.trial.Trial, search_space: dict) -> dict:
    '''Suggest hyperparameters from the given search space for a single Optuna trial.
    Args:
        trial: an Optuna trial object
        search_space: a dict of hyperparameter names to their search space spec, e.g.
            {
                "learning_rate": {"low": 1e-5, "high": 1e-3, "log": True},
                "weight_decay": {"low": 1e-6, "high": 1e-2, "log": True},
            }
    Returns:
        A dict of hyperparameter names to their suggested values for this trial.
    '''
    params = {}
    for name, spec in search_space.items():
        if "choices" in spec:
            params[name] = trial.suggest_categorical(name, spec["choices"])
        elif isinstance(spec.get("low"), int) and isinstance(spec.get("high"), int):
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        else:
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
    return params

def train(config: dict, run_id: str) -> str:
    """Single fixed-config training run — used when use_optuna is false.
    Args:
        config: the full config dict, including fixed_params
        run_id: the current run ID (e.g. a timestamp), used to name this run's subfolder
    Returns:
        path to the resulting checkpoint_best_total.pth
    """
    run_name = f"run_{run_id}"
    output_dir = str(Path(config["output_dir"]) / run_name)
    logger.info(f"Training single run (no Optuna) -> {output_dir}")
    return run_training(
        params=config["fixed_params"],
        dataset_dir=config["dataset_dir"],
        output_dir=output_dir,
        weighted_dataloader=config.get("weighted_dataloader", False),
        project_name=config.get("project_name", "textile-defect-detection"),
        run_name=run_name,
        pretrain_weights=config.get("pretrain_weights"),
    )


def train_with_optuna(config: dict, run_id: str) -> optuna.Study:
    """Multi-objective Optuna study over mAP50/recall/F1 — used when use_optuna is true.

    Each trial lands under <output_dir>/run_<run_id>/trial_<n>/ — the same
    output_dir as train(), so studies and single runs accumulate side by side.

    Args:
        config: the full config dict, including optuna.search_space and optuna.n_trials
        run_id: the current run ID (e.g. a timestamp), used to name this study's subfolder
    Returns:
        the best trial's output_dir (e.g. <output_dir>/run_<run_id>/trial_<n>/)
    """
    optuna_cfg = config["optuna"]
    run_name = f"run_{run_id}"

    def objective(trial: optuna.trial.Trial):
        params = _suggest_params(trial, optuna_cfg["search_space"])
        params["epochs"] = optuna_cfg["epochs"]
        params["batch_size"] = optuna_cfg["batch_size"]
        params["grad_accum_steps"] = optuna_cfg["grad_accum_steps"]

        trial_name = f"{run_name}/trial_{trial.number}"
        output_dir = str(Path(config["output_dir"]) / trial_name)
        run_training(
            params=params,
            dataset_dir=config["dataset_dir"],
            output_dir=output_dir,
            weighted_dataloader=config.get("weighted_dataloader", False),
            project_name=config.get("project_name", "textile-defect-detection"),
            run_name=trial_name,
            pretrain_weights=config.get("pretrain_weights"),
        )
        return _best_epoch_objectives(output_dir)

    study = optuna.create_study(
        study_name=f"rfdetr-tuning-{run_id}",
        directions=["maximize", "maximize", "maximize"],
    )
    study.optimize(objective, n_trials=optuna_cfg["n_trials"])

    for t in study.best_trials:
        logger.info(f"Trial {t.number}: mAP50={t.values[0]}, recall={t.values[1]}, F1={t.values[2]} params={t.params}")

    # return best folder 
    study_best_trial = study.best_trials[0]
    best_trial_folder = Path(config["output_dir"]) / f"{run_name}/trial_{study_best_trial.number}"
    logger.info(f"Best trial folder: {best_trial_folder}")
    return best_trial_folder
