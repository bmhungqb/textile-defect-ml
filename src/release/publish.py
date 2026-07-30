"""Publish the best model of a dataset version: push its artifacts to GCS and cut a
GitHub release.

The GCS layout is one folder per released version, named after the dataset version
it came from by default:

    gs://<bucket>/<prefix>/<version>/checkpoint_best_total.pth
                                    /metrics.csv
                                    /params.json
                                    /info.json                 # the dataset version trained on
                                    /label_studio_tasks.json   # its labels, as exported
                                    /experiments.md
                                    /experiments.csv
                                    /release.json

The GitHub release carries the dataset distribution and the experiment table as its
body, and the gs:// URIs as the download pointers. No files are attached to the
release — every artifact stays in GCS.
"""

import json
import os
import subprocess
from pathlib import Path

import pandas as pd

from src.release import summarize
from src.utils.gcs import upload_files
from src.utils.logger import get_logger

logger = get_logger(__name__)


def gcs_destination(bucket: str, prefix: str, version: str) -> str:
    """Build the gs:// prefix a version publishes to."""
    return f"gs://{bucket.strip('/')}/{prefix.strip('/')}/{version}"


def public_url(gs_uri: str) -> str:
    """Browser URL for a gs:// URI (viewable by anyone with bucket access)."""
    return gs_uri.replace("gs://", "https://storage.cloud.google.com/", 1)


def write_summary_files(df: pd.DataFrame, table_md: str, dest_dir: str) -> dict[str, str]:
    """Write experiments.csv / experiments.md into the dataset version folder.

    Args:
        df: table from summarize.build_table
        table_md: rendered Markdown table
        dest_dir: folder to write into (created if missing)
    Returns:
        {"csv": path, "md": path}
    """
    out = Path(dest_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path, md_path = out / "experiments.csv", out / "experiments.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(f"# Experiment summary\n\n{table_md}\n")
    return {"csv": str(csv_path), "md": str(md_path)}


def render_release_notes(
    version: str,
    info: dict,
    df: pd.DataFrame,
    best: pd.Series,
    table_md: str,
    uploaded: dict[str, str],
    metric: str,
    message: str | None = None,
) -> str:
    """Render the GitHub release body: message, headline metrics, dataset
    distribution, GCS links, and the full experiment comparison.

    Args:
        version: release version, e.g. v20260729_145230
        info: the dataset version manifest from data.build_dataset.load_info
        df: the full comparison table from summarize.build_table
        best: winning row from summarize.select_best
        table_md: rendered Markdown table of all experiments
        uploaded: {local_path: gs_uri} from the GCS upload
        metric: metric the winner was selected on
        message: optional release note from the user
    Returns:
        Markdown release notes
    """
    checkpoints = {Path(p).name: uri for p, uri in uploaded.items() if p.endswith(".pth")}
    weights_uri = checkpoints.get("checkpoint_best_total.pth", next(iter(checkpoints.values()), ""))
    dest = weights_uri.rsplit("/", 1)[0] if weights_uri else ""

    metrics_md = "\n".join(
        f"| {name} | {best[name]:.4f} |"
        for name in ("mAP50", "ema_mAP50", "recall", "F1")
        if name in best and pd.notna(best[name])
    )
    artifacts_md = "\n".join(
        f"| `{Path(p).name}` | `{uri}` |" for p, uri in sorted(uploaded.items(), key=lambda kv: kv[0])
    )
    message_md = f"\n> {message}\n" if message else ""

    return f"""## Model {version}
{message_md}
Dataset version `{info['version']}` · released from `{best['name']}`, the best of
{len(df)} run(s) across {df['experiment'].nunique()} experiment(s), selected by highest `{metric}`.

### Metrics (best epoch, validation)

| metric | value |
| --- | --- |
{metrics_md}

### Dataset version `{info['version']}`

Built {info['created_at']} from Label Studio project {info['source'].get('project_id')}.

{summarize.dataset_markdown(info)}

### Download

```bash
gsutil -m cp -r {dest} ./{version}
```

Weights: [`{weights_uri}`]({public_url(weights_uri)})

| artifact | gs:// URI |
| --- | --- |
{artifacts_md}

### Load

```python
from rfdetr import RFDETRMedium

model = RFDETRMedium(pretrain_weights="{version}/checkpoint_best_total.pth")
```

### All experiments

{table_md}
"""


def publish_to_gcs(
    run_dir: str,
    dest_uri: str,
    extra_files: list[str] | None = None,
    full: bool = False,
) -> dict[str, str]:
    """Upload the winning run's release artifacts to GCS.

    Args:
        run_dir: the winning run's folder (an experiment_* or trial_* directory)
        dest_uri: gs:// prefix from gcs_destination
        extra_files: version-level files to include (info.json, label_studio_tasks.json,
            experiments.csv/md, release.json)
        full: upload the whole run folder, not just RELEASE_FILES
    Returns:
        {local_path: gs_uri} for the files that uploaded successfully
    """
    paths = summarize.release_files(run_dir, full=full)
    if not paths:
        raise FileNotFoundError(f"No release artifacts found in {run_dir}")

    uploaded = upload_files(paths, dest_uri, base_dir=run_dir if full else None)
    if extra_files:
        uploaded |= upload_files(extra_files, dest_uri)
    return uploaded


def _repo_slug(remote: str = "origin") -> str:
    """Read <owner>/<repo> from a git remote URL."""
    url = subprocess.run(
        ["git", "remote", "get-url", remote],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return url.removesuffix(".git").removeprefix("git@github.com:").split("github.com/")[-1]


def create_github_release(
    tag: str,
    title: str,
    notes: str,
    repo: str | None = None,
    draft: bool = True,
) -> str:
    """Create a GitHub release via the gh CLI.

    The release carries no file attachments — every artifact lives in GCS and is
    referenced by its gs:// URI in the notes.

    Requires gh (https://cli.github.com) authenticated with repo write access.
    When gh is unavailable the notes are still written to disk by the caller, so
    the release can be created by hand from that file.

    Args:
        tag: release tag, e.g. model-v1.2.0
        title: release title
        notes: Markdown release body
        repo: <owner>/<repo>; read from the origin remote when omitted
        draft: create as a draft so it can be reviewed before publishing
    Returns:
        the release URL
    """
    repo = repo or _repo_slug()
    notes_file = Path(os.environ.get("TMPDIR", "/tmp")) / f"release_notes_{tag}.md"
    notes_file.write_text(notes)

    cmd = ["gh", "release", "create", tag, "--repo", repo, "--title", title,
           "--notes-file", str(notes_file)]
    if draft:
        cmd.append("--draft")

    logger.info(f"Creating GitHub release {tag} on {repo} (draft={draft})")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError(
            "gh CLI not found. Install it (brew install gh && gh auth login), or re-run with "
            "--no-github and create the release by hand from the generated RELEASE_NOTES.md."
        ) from None
    if result.returncode != 0:
        raise RuntimeError(f"gh release create failed: {result.stderr.strip()}")

    url = result.stdout.strip()
    logger.info(f"Release created: {url}")
    return url


def write_manifest(dest_dir: str, manifest: dict) -> str:
    """Write release.json — the machine-readable record of what was published."""
    path = Path(dest_dir) / "release.json"
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return str(path)
