"""GCS connection helpers

Authentication comes from a service-account JSON key file, pointed at by
GOOGLE_APPLICATION_CREDENTIALS in .env (see get_client). A relative path there is
resolved against the project root, so the pipelines work from any working directory.
Without that variable the client falls back to Application Default Credentials — the
attached service account when running on a GCE/GKE instance.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from google.cloud import storage
from src.utils.logger import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a gs:// URI: {uri}")
    bucket_name, blob_path = uri.replace("gs://", "").split("/", 1)
    return bucket_name, blob_path


def resolve_credentials(credentials: str | None = None) -> str | None:
    """Locate the service-account key file to authenticate with.

    Args:
        credentials: explicit path to a service-account JSON key; falls back to
            GOOGLE_APPLICATION_CREDENTIALS
    Returns:
        an absolute path to the key file, or None to use Application Default
        Credentials
    Raises:
        FileNotFoundError: a path was configured but nothing is there — better to
            fail loudly than to silently fall back to ADC and get a 403 later
    """
    path = credentials or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not path:
        return None

    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = _PROJECT_ROOT / resolved
    if not resolved.exists():
        raise FileNotFoundError(
            f"GCS credentials file not found: {resolved} (from "
            f"{'the credentials argument' if credentials else 'GOOGLE_APPLICATION_CREDENTIALS'}). "
            "Point it at the service-account JSON key on this machine, or unset it to use "
            "the instance's default credentials."
        )
    return str(resolved)


@lru_cache(maxsize=None)
def get_client(credentials: str | None = None) -> storage.Client:
    """Build the GCS client — cached, so one client is shared across transfers.

    Args:
        credentials: explicit path to a service-account JSON key; defaults to
            GOOGLE_APPLICATION_CREDENTIALS
    Returns:
        an authenticated storage.Client
    """
    key_file = resolve_credentials(credentials)
    if key_file:
        logger.info(f"Authenticating to GCS with service account {key_file}")
        return storage.Client.from_service_account_json(key_file)

    logger.info("No GOOGLE_APPLICATION_CREDENTIALS set — using Application Default Credentials")
    return storage.Client()

def download_blobs(
    uris: list[str],
    dest_dir: str,
    workers: int = 16,
    credentials: str | None = None,
) -> dict[str, str]:
    """Download gs:// URIs into dest_dir.
    Args:
        uris: list of gs:// URIs to download
        dest_dir: local directory to download into
        workers: number of threads to use for parallel downloads
        credentials: path to a service-account JSON key (see get_client)
    Returns:
        {uri: local_file_name} for the URIs that downloaded successfully
    """
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    client = get_client(credentials)
    bucket_cache: dict[str, storage.Bucket] = {}

    def _download_one(uri: str) -> tuple[str, str]:
        bucket_name, blob_path = parse_gs_uri(uri)
        if bucket_name not in bucket_cache:
            bucket_cache[bucket_name] = client.bucket(bucket_name)
        local_name = Path(blob_path).name
        bucket_cache[bucket_name].blob(blob_path).download_to_filename(str(dest_path / local_name))
        return uri, local_name

    downloaded: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download_one, uri): uri for uri in uris}
        for i, future in enumerate(as_completed(futures), 1):
            uri = futures[future]
            try:
                uri, local_name = future.result()
                downloaded[uri] = local_name
            except Exception as e:
                logger.warning(f"Failed to download {uri}: {e}")
            if i % 200 == 0:
                logger.info(f"...{i}/{len(uris)} downloaded")

    logger.info(f"Downloaded {len(downloaded)}/{len(uris)} blobs to {dest_dir}")
    return downloaded


def upload_files(
    paths: list[str],
    dest_uri: str,
    base_dir: str | None = None,
    workers: int = 8,
    credentials: str | None = None,
) -> dict[str, str]:
    """Upload local files under a gs:// prefix.

    Args:
        paths: local file paths to upload
        dest_uri: destination prefix, e.g. gs://bucket/models/textile/v1.0.0
        base_dir: if given, each file keeps its path relative to base_dir under
            dest_uri; otherwise files land flat under dest_uri by file name
        workers: number of threads to use for parallel uploads
        credentials: path to a service-account JSON key (see get_client)
    Returns:
        {local_path: gs_uri} for the files that uploaded successfully
    """
    bucket_name, prefix = parse_gs_uri(dest_uri.rstrip("/"))
    bucket = get_client(credentials).bucket(bucket_name)

    def _upload_one(path: str) -> tuple[str, str]:
        local = Path(path)
        rel = local.relative_to(base_dir).as_posix() if base_dir else local.name
        blob_path = f"{prefix}/{rel}"
        bucket.blob(blob_path).upload_from_filename(str(local))
        return path, f"gs://{bucket_name}/{blob_path}"

    uploaded: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_upload_one, p): p for p in paths}
        for future in as_completed(futures):
            path = futures[future]
            try:
                path, uri = future.result()
                uploaded[path] = uri
                logger.info(f"Uploaded {path} -> {uri}")
            except Exception as e:
                logger.warning(f"Failed to upload {path}: {e}")

    logger.info(f"Uploaded {len(uploaded)}/{len(paths)} files to {dest_uri}")
    return uploaded
