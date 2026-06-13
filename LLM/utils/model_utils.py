import os
from pathlib import Path


def resolve_model_source(model_name: str, local_model_root: str | None = None):
    """Prefer a matching model under the repo's model/ directory before remote download."""
    if not model_name:
        return model_name, False

    model_path = Path(model_name).expanduser()
    if model_path.exists():
        return str(model_path), False

    if local_model_root is None:
        local_model_root = Path(__file__).resolve().parents[2] / "model"
    else:
        local_model_root = Path(local_model_root).expanduser()

    local_model_root = local_model_root.resolve()
    if not local_model_root.exists():
        return model_name, False

    search_names = []
    normalized_name = model_name.strip("/").replace("\\", "/")
    if normalized_name:
        search_names.append(normalized_name.split("/")[-1])
        search_names.append(normalized_name.replace("/", os.sep))

    seen = set()
    for search_name in search_names:
        if not search_name or search_name in seen:
            continue
        seen.add(search_name)

        direct_candidate = local_model_root / search_name
        if direct_candidate.exists():
            return str(direct_candidate), True

        nested_candidate = local_model_root / Path(search_name)
        if nested_candidate.exists():
            return str(nested_candidate), True

    for candidate in local_model_root.iterdir():
        if candidate.is_dir() and candidate.name == search_names[0]:
            return str(candidate), True

    return model_name, False
