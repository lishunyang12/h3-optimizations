"""Diagnostic-only unsynchronized NVTX scopes; no model math or stream edits."""
from contextlib import contextmanager
from functools import wraps
import os

_ENABLED = os.environ.get("H3_PROFILE_NVTX", "0") == "1"

@contextmanager
def profile_range(name):
    if not _ENABLED:
        yield
        return
    import nvtx
    nvtx.push_range("h3.profile." + name, domain="h3.profile")
    try:
        yield
    finally:
        nvtx.pop_range(domain="h3.profile")

def profile_method(name):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with profile_range(name):
                return function(*args, **kwargs)
        return wrapped
    return decorate
