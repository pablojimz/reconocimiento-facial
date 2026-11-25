"""Scoped warning filters for upstream models."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def quiet_model_warnings() -> Iterator[None]:
    """Suppress only upstream model warnings inside the managed operation."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", module=r"^(torch|torchvision|timm|transface_vit)(\.|$)"
        )
        yield
