from pathlib import Path
import yaml


def read_yaml_file(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"File {path} does not exist")

    if not path.endswith(".yaml"):
        raise ValueError(f"File {path} is not a yaml file")

    with open(path, "r") as f:
        return yaml.safe_load(f)
