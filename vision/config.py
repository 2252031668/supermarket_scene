"""Read local provider credentials without exposing them in tracked source files."""

from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


VISION_DIR = Path(__file__).resolve().parent
LOCAL_CONFIG_PATH = VISION_DIR / "config.local.yaml"
DEFAULT_INSPECTION_CONFIG = {
    "min_current_coverage": 0.05,
    "analysis_center_ratio": 0.8,
    "lab_distance_threshold": 12.0,
    "slot_change_ratio_threshold": 0.15,
    "dino_confidence_threshold": 0.72,
    "ambiguity_margin": 0.05,
    "vlm_fallback": False,
    "vlm_top_k": 4,
}
DEFAULT_SKU_QUERY_CONFIG = {
    "max_boxes": 1,
    "dino_fallback": False,
    "dino_confidence_threshold": 0.72,
}


@lru_cache(maxsize=1)
def local_config() -> dict[str, Any]:
    if not LOCAL_CONFIG_PATH.is_file():
        return {}
    loaded: Any = yaml.safe_load(LOCAL_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise RuntimeError(str(LOCAL_CONFIG_PATH) + " must contain a YAML mapping")
    return loaded


@lru_cache(maxsize=1)
def local_api_keys() -> dict[str, str]:
    loaded = local_config()
    raw_keys = loaded.get("api_keys", {})
    if not isinstance(raw_keys, dict):
        raise RuntimeError(str(LOCAL_CONFIG_PATH) + ": api_keys must be a YAML mapping")
    return {
        str(provider): value.strip()
        for provider, value in raw_keys.items()
        if isinstance(value, str) and value.strip()
    }


def get_inspection_config() -> dict[str, Any]:
    """Return saved inspection options merged with stable defaults."""
    loaded = local_config().get("inspection", {})
    if not isinstance(loaded, dict):
        raise RuntimeError(str(LOCAL_CONFIG_PATH) + ": inspection must be a YAML mapping")
    # save_debug belonged to the old persisted UI setting. Debug is now chosen
    # by each caller because it changes execution work rather than recognition.
    loaded = {key: value for key, value in loaded.items() if key != "save_debug"}
    return {**DEFAULT_INSPECTION_CONFIG, **loaded}


def save_inspection_config(inspection: dict[str, Any]) -> dict[str, Any]:
    """Persist inspection options without replacing provider keys."""
    merged = {**DEFAULT_INSPECTION_CONFIG, **inspection}
    loaded = dict(local_config())
    loaded["inspection"] = merged
    LOCAL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=LOCAL_CONFIG_PATH.parent, delete=False
    ) as temporary:
        yaml.safe_dump(loaded, temporary, allow_unicode=True, sort_keys=False)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, LOCAL_CONFIG_PATH)
    local_config.cache_clear()
    local_api_keys.cache_clear()
    return merged


def get_sku_query_config() -> dict[str, Any]:
    """Return saved SKU-query options merged with stable defaults."""
    loaded = local_config().get("sku_query", {})
    if not isinstance(loaded, dict):
        raise RuntimeError(str(LOCAL_CONFIG_PATH) + ": sku_query must be a YAML mapping")
    return {**DEFAULT_SKU_QUERY_CONFIG, **loaded}


def save_sku_query_config(sku_query: dict[str, Any]) -> dict[str, Any]:
    """Persist SKU-query options without replacing provider keys or inspection settings."""
    merged = {**DEFAULT_SKU_QUERY_CONFIG, **sku_query}
    loaded = dict(local_config())
    loaded["sku_query"] = merged
    LOCAL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=LOCAL_CONFIG_PATH.parent, delete=False
    ) as temporary:
        yaml.safe_dump(loaded, temporary, allow_unicode=True, sort_keys=False)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, LOCAL_CONFIG_PATH)
    local_config.cache_clear()
    local_api_keys.cache_clear()
    return merged


def get_api_key(provider: str, environment_variable: str) -> str | None:
    """Prefer the ignored local YAML file, with environment variables as a fallback."""
    return local_api_keys().get(provider) or os.environ.get(environment_variable)


def require_api_key(provider: str, environment_variable: str) -> str:
    api_key = get_api_key(provider, environment_variable)
    if api_key:
        return api_key
    raise RuntimeError(
        f"No {provider} API key is configured. Add it to {LOCAL_CONFIG_PATH} "
        f"or set {environment_variable}."
    )
