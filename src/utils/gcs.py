"""GCS connection helpers
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from google.cloud import storage
from src.utils.logger import get_logger

logger = get_logger(__name__)

def parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a gs:// URI: {uri}")
    bucket_name, blob_path = uri.replace("gs://", "").split("/", 1)
    return bucket_name, blob_path

def download_blobs(uris: list[str], dest_dir: str, workers: int = 16) -> dict[str, str]:
    """Download gs:// URIs into dest_dir.
    Args:
        uris: list of gs:// URIs to download
        dest_dir: local directory to download into
        workers: number of threads to use for parallel downloads
    Returns:
        {uri: local_file_name} for the URIs that downloaded successfully
    """
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    client = storage.Client()
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
