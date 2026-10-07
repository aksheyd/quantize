"""Temporary: reports, as a warning, how long another thread pauses while
alternate refits, and how long a 1 ms sleep lasts on this machine."""

import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from quantize import learned, quantize


def longest_pause_during_alternate(quantized, weights):
    longest_pause = 0.0
    start = last_check = time.perf_counter()
    with ThreadPoolExecutor() as pool:
        refitting = pool.submit(learned.alternate, quantized, weights)
        while not refitting.done():
            now = time.perf_counter()
            longest_pause = max(longest_pause, now - last_check)
            last_check = now
    refitting.result()
    return longest_pause, last_check - start


def sleeps_during_alternate(quantized, weights):
    wakeups = [time.perf_counter()]
    with ThreadPoolExecutor() as pool:
        refitting = pool.submit(learned.alternate, quantized, weights)
        while not refitting.done():
            time.sleep(0.001)
            wakeups.append(time.perf_counter())
    refitting.result()
    return np.diff(wakeups), wakeups[-1] - wakeups[0]


def test_report_pauses_during_alternate():
    weights = np.random.default_rng(0).standard_normal((1024, 1024)).astype(np.float32)
    pauses, refit_seconds, sleep_seconds, old_bar_passes = [], [], [], 0
    deadline = time.perf_counter() + 8
    while len(pauses) < 40 and time.perf_counter() < deadline:
        pause, refit = longest_pause_during_alternate(quantize(weights, bits=4), weights)
        pauses.append(pause)
        refit_seconds.append(refit)
        sleeps, sleeping_refit = sleeps_during_alternate(quantize(weights, bits=4), weights)
        sleep_seconds.extend(sleeps)
        old_bar_passes += len(sleeps) > sleeping_refit / 0.010
    shares = np.array(pauses) / np.array(refit_seconds)
    milliseconds = 1e3
    warnings.warn(
        f"{len(pauses)} refits, median {np.median(refit_seconds) * milliseconds:.1f} ms"
        f" | longest pause median {np.median(pauses) * milliseconds:.2f} ms,"
        f" max {max(pauses) * milliseconds:.2f} ms"
        f" | pause / refit median {np.median(shares):.3f}, max {shares.max():.3f}"
        f" | 1 ms sleep median {np.median(sleep_seconds) * milliseconds:.2f} ms,"
        f" 90th percentile {np.percentile(sleep_seconds, 90) * milliseconds:.2f} ms,"
        f" max {max(sleep_seconds) * milliseconds:.2f} ms"
        f" | old wake-up bar passed {old_bar_passes} of {len(pauses)}"
    )
