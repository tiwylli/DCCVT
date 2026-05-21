"""DCCVT package exports."""


def run_from_args_file(*args, **kwargs):
    """Run experiments from an args file, importing the CUDA runtime only when needed."""
    from dccvt.api import run_from_args_file as _run_from_args_file

    return _run_from_args_file(*args, **kwargs)


def run_mesh_from_params(*args, **kwargs):
    """Run one DCCVT mesh experiment, importing the CUDA runtime only when needed."""
    from dccvt.api import run_mesh_from_params as _run_mesh_from_params

    return _run_mesh_from_params(*args, **kwargs)

__all__ = [
    "run_from_args_file",
    "run_mesh_from_params",
]
