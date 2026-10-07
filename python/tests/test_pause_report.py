"""Temporary: report the alternate pause test's numbers on each platform."""

import sys
import sysconfig
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from quantize import learned, quantize


def measure(weights):
    quantized = quantize(weights, bits=4)
    longest_pause = 0.0
    start = last_check = time.perf_counter()
    with ThreadPoolExecutor() as pool:
        refitting = pool.submit(learned.alternate, quantized, weights)
        while not refitting.done():
            now = time.perf_counter()
            longest_pause = max(longest_pause, now - last_check)
            last_check = now
    refitting.result()
    return last_check - start, longest_pause


def test_pause_report():
    gil = "GIL off" if sysconfig.get_config_var("Py_GIL_DISABLED") and not sys._is_gil_enabled() else "GIL on"
    lines = [f"Python {sys.version.split()[0]} {gil}"]
    for side in [1024, 2048]:
        began = time.perf_counter()
        weights = np.random.default_rng(0).standard_normal((side, side)).astype(np.float32)
        refits, pauses = [], []
        while len(refits) < 12 and time.perf_counter() - began < 6:
            refit, pause = measure(weights)
            refits.append(refit)
            pauses.append(pause)
        refits, pauses = np.array(refits), np.array(pauses)
        lines.append(
            f"{side}x{side}: {len(refits)} refits, refit median {np.median(refits) * 1e3:.0f} ms"
            f" min {refits.min() * 1e3:.0f} max {refits.max() * 1e3:.0f}"
            f" | longest pause median {np.median(pauses) * 1e3:.2f} ms max {pauses.max() * 1e3:.2f}"
            f" | pause/refit max {(pauses / refits).max():.3f}"
            f" | {(time.perf_counter() - began) / len(refits) * 1e3:.0f} ms per test body"
        )
    warnings.warn("PAUSE REPORT " + " || ".join(lines))
