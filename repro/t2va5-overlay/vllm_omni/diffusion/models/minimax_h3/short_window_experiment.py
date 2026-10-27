"""Explicit experimental shapes; legacy paths keep their original checks."""
import os

def enabled():
    return os.environ.get('H3_EXPERIMENT_SHORT_WINDOWS') == '1'

def windows():
    return (56, 73, 124, 141, 362) if enabled() else (124, 141, 362)

def latents():
    return (17, 22, 37, 42, 107) if enabled() else (37, 42, 107)

def short_request(task, extra):
    options = extra.get('rolling_continuation')
    return (enabled() and task == 'ref2va' and isinstance(options, dict)
            and type(options.get('window_frames')) is int
            and options['window_frames'] in (56, 73)
            and options.get('overlap_frames') == 5)
