"""Utility functions for input/output operations."""

import time
from pathlib import Path
from typing import Any

import yaml


def get_yaml_dict(yaml_obj: dict[str, Any] | Path | str) -> dict[str, Any]:
    """
    Get a dictionary from a YAML object or file.

    Parameters
    ----------
    yaml_obj : dict | Path | str
        A dictionary or a path to a YAML file.

    Returns
    -------
    dict[str, Any]
        The dictionary obtained from the YAML file, or the input dictionary itself.
    """
    if isinstance(yaml_obj, dict):
        return yaml_obj
    with Path(yaml_obj).open() as stream:
        yaml_obj = yaml.safe_load(stream)
        if not isinstance(yaml_obj, dict):
            raise ValueError(f"The YAML file {yaml_obj} does not contain a valid dictionary.")
        return yaml_obj


def purge_files(
    path: Path | str,
    pattern: str,
    older_than: float | None = None,
    more_than: int | None = None,
    must_both_be_true: bool = False,
    recursive: bool = False,
    safe: bool = True,
    excludes: list[str] | None = None,
) -> list[str]:
    """
    Purge files in a directory based on age and/or quantity criteria.

    Parameters
    ----------
    path : Path | str
        The directory path where files are located.
    pattern : str
        The glob pattern to match files.
    older_than : float, optional
        Age in seconds; files older than this will be purged.
    more_than : int, optional
        If the number of matching files exceeds this number, the oldest files will be purged.
    must_both_be_true : bool
        If True, files must meet both criteria to be purged; if False, meeting either criterion is sufficient.
    recursive : bool
        If True, search for files recursively in subdirectories.
    safe : bool
        If True, enables safe mode which restricts certain operations.
    excludes : list[str], optional
        List of file paths to exclude from purging.

    Returns
    -------
    list[str]
        List of file paths that were purged.
    """
    excludes = [] if excludes is None else excludes
    if safe:
        # safe mode prevents absolute path, and '*' pattern
        if not Path(path).is_absolute():
            raise ValueError("Path must be absolute in safe mode.")
        if pattern == "*":
            raise ValueError("Pattern must not be '*' in safe mode.")
    if recursive:
        matching_files = list(Path(path).rglob(pattern))
    else:
        matching_files = list(Path(path).glob(pattern))
    purge_candidates = []
    if older_than is not None:
        now = time.time()
        purge_candidates = [f for f in matching_files if f.stat().st_mtime < (now - older_than)]
    if (more_than is not None) and (len(matching_files) > more_than):
        matching_files = sorted(matching_files, key=lambda f: f.stat().st_mtime_ns, reverse=True)
        if purge_candidates and must_both_be_true:
            purge_candidates = [f for f in purge_candidates if f in matching_files[more_than:]]
        else:
            for matching_file in matching_files[more_than:]:
                if matching_file not in purge_candidates:
                    purge_candidates.append(matching_file)
    purged_files = []
    for f in purge_candidates:
        if str(f) in excludes:
            continue
        try:
            f.unlink()
        except OSError as e:
            print(f"Error deleting file {f}: {e}")
        else:
            purged_files.append(f)
    return [str(f) for f in purged_files]
