"""Label Studio connection helpers
"""

from label_studio_sdk import Client
from src.utils.logger import get_logger

logger = get_logger(__name__)

def get_client(url: str, api_key: str) -> Client:
    return Client(url, api_key)

def pull_tasks(url: str, api_key: str, project_id: int) -> list[dict]:
    """Pull every task of a Label Studio project as raw task dicts."""
    project = get_client(url, api_key).get_project(project_id)
    tasks = project.get_tasks()
    logger.info(f"Pulled {len(tasks)} tasks from Label Studio project {project_id}")
    return tasks
