import os
from pathlib import Path
from dotenv import load_dotenv
from __future__ import annotations

def load_env(dotenv_path: str | None = None) -> None:
  
    load_dotenv(dotenv_path)

def get_key(name: str, required: bool = True, default: str | None = None) -> str | None:

    val = os.getenv(name, default)
    if required and (val is None or str(val).strip() == ""):
        raise KeyError(f"Missing required environment variable: {name}")
    return val

def data_dir() -> Path:

    return Path(os.getenv("DATA_DIR", "./data")).expanduser().resolve()
