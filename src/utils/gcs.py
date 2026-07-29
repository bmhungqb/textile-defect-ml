"""Download images referenced by gs:// URIs to a local directory.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google.cloud import storage

from utils.logger import get_logger

logger = get_logger(__name__)


def parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a gs:// URI: {uri}")
    bucket_name, blob_path = uri.replace("gs://", "").split("/", 1)
    return bucket_name, blob_path


def download_images(images: list[dict], dest_dir: str, workers: int = 16) -> dict[int, str]:
    """Download images (COCO image dicts with gs:// file_name) into dest_dir.

    Returns {image_id: local_file_name} for images that downloaded successfully;
    failed downloads are logged and dropped from the result.
    """
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    client = storage.Client()
    bucket_cache: dict[str, storage.Bucket] = {}

    def _download_one(image: dict) -> tuple[int, str]:
        bucket_name, blob_path = parse_gs_uri(image["file_name"])
        if bucket_name not in bucket_cache:
            bucket_cache[bucket_name] = client.bucket(bucket_name)
        local_name = Path(blob_path).name
        bucket_cache[bucket_name].blob(blob_path).download_to_filename(str(dest_path / local_name))
        return image["id"], local_name

    downloaded: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download_one, img): img for img in images}
        for i, future in enumerate(as_completed(futures), 1):
            img = futures[future]
            try:
                image_id, local_name = future.result()
                downloaded[image_id] = local_name
            except Exception as e:
                logger.warning(f"Failed to download {img['file_name']}: {e}")
            if i % 200 == 0:
                logger.info(f"...{i}/{len(images)} downloaded")

    logger.info(f"Downloaded {len(downloaded)}/{len(images)} images to {dest_dir}")
    return downloaded
