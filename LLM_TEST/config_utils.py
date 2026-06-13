import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


ENV_VAR_PATTERN = re.compile(r"\$(?:\{([^}]+)\}|([A-Za-z_][A-Za-z0-9_]*))")


def load_env_file(env_path: Optional[str]) -> Dict[str, str]:
    if not env_path:
        return {}

    path = Path(env_path).resolve()
    if not path.exists():
        return {}

    loaded: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]

            loaded[key] = value
            os.environ[key] = value

    return loaded


def _expand_env_in_string(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        env_name = match.group(1) or match.group(2)
        env_value = os.environ.get(env_name)
        if env_value is None:
            raise KeyError(f"Environment variable '{env_name}' is not set.")
        return env_value

    return ENV_VAR_PATTERN.sub(replace, value)


def expand_env_vars(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: expand_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    if isinstance(value, str):
        return _expand_env_in_string(value) if "$" in value else value
    return value


def load_yaml_config(config_path: Optional[str]) -> Dict[str, Any]:
    if not config_path:
        return {}

    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML content must be a mapping: {path}")
    return expand_env_vars(data)


def get_section(config: Dict[str, Any], section: str) -> Dict[str, Any]:
    value = config.get(section, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{section}' must be a mapping.")
    return value


def resolve_value(cli_value: Any, section: Dict[str, Any], key: str, default: Any = None) -> Any:
    if cli_value not in (None, ""):
        return cli_value
    if key in section and section[key] not in (None, ""):
        return section[key]
    return default
