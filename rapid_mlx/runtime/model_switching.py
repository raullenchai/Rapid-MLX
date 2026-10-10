"""Shared HTTP contract for temporary primary-model/worker handoffs."""

from fastapi import HTTPException


def primary_switching() -> bool:
    """Read transition state without taking the residency loader's lock."""
    from ..config import get_config

    manager = get_config().residency_manager
    return getattr(manager, "primary_switching", False) is True


class ModelSwitchingError(HTTPException):
    """A new request can retry after the model-worker transition finishes."""

    def __init__(self) -> None:
        super().__init__(
            status_code=503,
            headers={"Retry-After": "5"},
            detail={
                "error": {
                    "message": "Model switching in progress. Retry after the transition completes.",
                    "type": "server_error",
                    "code": "model_switching",
                    "param": None,
                }
            },
        )
