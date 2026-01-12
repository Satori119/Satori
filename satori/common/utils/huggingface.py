
from typing import Optional
from satori.common.utils.logging import setup_logger

logger = setup_logger(__name__)


def parse_dataset_url(url: str) -> str:

    if not url:
        return ""

    if url.startswith("https://huggingface.co/datasets/"):
        dataset_id = url.replace("https://huggingface.co/datasets/", "")
    elif url.startswith("http://huggingface.co/datasets/"):
        dataset_id = url.replace("http://huggingface.co/datasets/", "")
    elif "/" in url and not url.startswith("http"):
        dataset_id = url
    else:
        return url

    if "?" in dataset_id:
        dataset_id = dataset_id.split("?")[0]
    if "#" in dataset_id:
        dataset_id = dataset_id.split("#")[0]

    return dataset_id.strip("/")


def parse_model_url(url: str) -> str:

    if not url:
        return ""

    if url.startswith("https://huggingface.co/"):
        model_id = url.replace("https://huggingface.co/", "")
    elif url.startswith("http://huggingface.co/"):
        model_id = url.replace("http://huggingface.co/", "")
    elif "/" in url and not url.startswith("http"):
        model_id = url
    else:
        return url

    if model_id.startswith("datasets/"):
        model_id = model_id.replace("datasets/", "", 1)

    if "?" in model_id:
        model_id = model_id.split("?")[0]
    if "#" in model_id:
        model_id = model_id.split("#")[0]

    return model_id.strip("/")


def build_dataset_url(repository_id: str) -> str:

    if not repository_id:
        return ""

    repository_id = repository_id.strip("/")
    return f"https://huggingface.co/datasets/{repository_id}"


def build_model_url(repository_id: str) -> str:

    if not repository_id:
        return ""

    repository_id = repository_id.strip("/")
    return f"https://huggingface.co/{repository_id}"


def is_valid_hf_repository_id(repository_id: str) -> bool:

    if not repository_id:
        return False

    parts = repository_id.strip("/").split("/")
    if len(parts) < 2:
        return False

    return all(part.strip() for part in parts[:2])
