"""Phase 3 entrypoint: release the best model of a dataset version.

Takes a dataset version folder — output/<version>/, holding info.json and every
experiment trained against it — and:

1. Summarizes every run under it (each experiment's Optuna trials, or its single
   fixed-config run) into one comparison table: name, hyperparameters, metrics.
2. Picks the best run — highest F1 among runs that produced a checkpoint — or the
   explicit --index.
3. Uploads that run's artifacts, the dataset info.json and the summary table to
   gs://<bucket>/<prefix>/<version>/.
4. Cuts a draft GitHub release whose body is the message, the dataset distribution,
   the gs:// links and the comparison table. Nothing is attached to the release
   itself — the artifacts stay in GCS and are referenced by link.

    python pipelines/run_release_pipeline.py output/v20260729_145230 -m "first release"
"""

import argparse
import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.data.build_dataset import load_info
from src.release import publish, summarize
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "data_version_dir",
        help="Dataset version folder to release from, e.g. output/v20260729_145230",
    )
    parser.add_argument("-m", "--message", help="Release note shown at the top of the GitHub release")
    parser.add_argument("--config", default="configs/release.yaml")
    parser.add_argument(
        "--version",
        help="Release version (default: the dataset version). Pass one explicitly when "
        "re-releasing the same dataset version after more experiments, e.g. v20260729_145230-r2",
    )
    parser.add_argument("--index", type=int, help="Release this row of the table instead of the best-scoring run")
    parser.add_argument("--select-by", help="Metric to pick the best run on (overrides the config)")
    parser.add_argument("--bucket", help="GCS bucket (overrides the config)")
    parser.add_argument("--full", action="store_true", help="Upload the whole run folder, including checkpoint.pth")
    parser.add_argument("--publish", action="store_true", help="Publish the GitHub release instead of leaving it a draft")
    parser.add_argument("--no-github", action="store_true", help="Upload to GCS only; write the notes to disk without calling gh")
    parser.add_argument("--dry-run", action="store_true", help="Build the table and notes, upload nothing, release nothing")
    args = parser.parse_args()

    # GOOGLE_APPLICATION_CREDENTIALS lives in .env, and the GCS client reads it from
    # the environment.
    load_dotenv()

    config = load_yaml(args.config)
    gcs_cfg, gh_cfg = config.get("gcs", {}), config.get("github", {})
    bucket = args.bucket or gcs_cfg.get("bucket")
    metric = args.select_by or config.get("select_by", "F1")

    info = load_info(args.data_version_dir)
    version = args.version or info["version"]

    # 1. Compare every experiment run against this dataset version.
    df = summarize.build_table(args.data_version_dir)

    # 2. Pick what to release.
    if args.index is not None:
        matches = df[df["index"] == args.index]
        if matches.empty:
            raise SystemExit(f"No run with index {args.index} under {args.data_version_dir}")
        best = matches.iloc[0]
    else:
        best = summarize.select_best(df, metric=metric)

    table_md = summarize.to_markdown(df, best_index=int(best["index"]), metric=metric)
    summary_paths = publish.write_summary_files(df, table_md, args.data_version_dir)
    logger.info(f"Experiment summary written to {summary_paths['md']}")

    if not bucket:
        raise SystemExit("No GCS bucket configured — set gcs.bucket in the config or pass --bucket")
    dest_uri = publish.gcs_destination(bucket, gcs_cfg.get("prefix", "models"), version)
    full = args.full or config.get("upload", {}).get("full", False)

    if args.dry_run:
        would_upload = summarize.release_files(str(best["path"]), full=full)
        preview = {p: f"{dest_uri}/{Path(p).name}" for p in would_upload}
        print(publish.render_release_notes(version, info, df, best, table_md, preview, metric, args.message))
        logger.info(f"Dry run — would upload {len(would_upload)} file(s) from {best['path']} to {dest_uri}")
        return

    # 3. Upload the winning run's artifacts, alongside the dataset manifest and table.
    uploaded = publish.publish_to_gcs(
        run_dir=str(best["path"]),
        dest_uri=dest_uri,
        extra_files=[
            str(Path(args.data_version_dir) / "info.json"),
            summary_paths["csv"],
            summary_paths["md"],
        ],
        full=full,
    )

    notes = publish.render_release_notes(version, info, df, best, table_md, uploaded, metric, args.message)
    manifest_path = publish.write_manifest(
        args.data_version_dir,
        {
            "version": version,
            "message": args.message,
            "data_version": info["version"],
            "data_version_dir": args.data_version_dir,
            "run": best["name"],
            "selected_by": metric,
            "metrics": {k: best[k] for k in ("mAP50", "ema_mAP50", "recall", "F1") if k in best},
            "gcs_uri": dest_uri,
            "artifacts": uploaded,
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        },
    )
    publish.upload_files([manifest_path], dest_uri)

    # 4. Cut the GitHub release.
    notes_path = f"{args.data_version_dir}/RELEASE_NOTES.md"
    with open(notes_path, "w") as f:
        f.write(notes)

    if args.no_github:
        logger.info(f"Skipping GitHub release. Notes: {notes_path}")
    else:
        url = publish.create_github_release(
            tag=f"{gh_cfg.get('tag_prefix', 'model-')}{version}",
            title=f"Model {version} — {best['name']} ({metric}={best[metric]:.4f})",
            notes=notes,
            repo=gh_cfg.get("repo") or None,
            draft=not args.publish and gh_cfg.get("draft", True),
        )
        logger.info(f"GitHub release: {url}")

    logger.info(f"Done. {best['name']} released as {version} -> {dest_uri}")


if __name__ == "__main__":
    main()
