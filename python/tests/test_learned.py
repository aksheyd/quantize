import sys
import sysconfig
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from quantize import (
    LengthMismatchError,
    Quantized,
    QuantizeError,
    adaptive,
    asymmetric,
    learned,
    quantize,
    quantize_tensor,
)


def test_fit_recovers_known_line():
    codes = [0, 1, 2, 3, 4]
    values = [0.5 * code + 1.0 for code in codes]
    scale, zero_point = learned.fit_scale_and_zero_point(values, codes)
    assert abs(scale - 0.5) < 1e-5
    assert abs(zero_point + 2.0) < 1e-5


def test_refine_keeps_kind_and_identity():
    weights = [0.42, -0.10, 0.70, -0.50]
    quantized = quantize(weights, bits=8, block=4)
    before = quantized.copy()
    result = learned.refine(quantized, weights)
    assert result is quantized
    assert quantized.kind == "symmetric"
    assert quantized.nbytes == before.nbytes
    np.testing.assert_array_equal(quantized.unpacked_codes, before.unpacked_codes)


def test_refine_takes_the_matrix_it_refines():
    weights = np.linspace(-0.5, 0.5, 64, dtype=np.float32).reshape(2, 32)
    quantized = quantize(weights, bits=4, block=32)
    learned.refine(quantized, weights)
    assert quantized.shape == (2, 32)


def test_refine_and_alternate_need_values_in_the_tensor_shape():
    weights = np.linspace(-0.5, 0.5, 64, dtype=np.float32).reshape(2, 32)
    quantized = quantize(weights, bits=4)
    before = quantized.copy()
    for refit in [learned.refine, learned.alternate]:
        for values in [weights.T, weights.ravel()]:
            with pytest.raises(ValueError, match=r"values must have shape \(2, 32\), got"):
                refit(quantized, values)
        with pytest.raises(LengthMismatchError, match="values must have length 64, got 32"):
            refit(quantized, weights[:1])
    assert quantized == before
    with pytest.raises(ValueError, match=r"values must have shape \(64,\), got \(2, 32\)"):
        learned.refine(quantize(weights.ravel()), weights)


def test_refine_length_mismatch_including_empty():
    quantized = quantize([0.1] * 4, bits=8, block=4)
    with pytest.raises(LengthMismatchError):
        learned.refine(quantized, [0.1] * 3)
    empty = quantize([], bits=8, block=4)
    with pytest.raises(LengthMismatchError):
        learned.refine(empty, [0.1])
    assert learned.refine(empty, []) is empty


def test_alternate_moves_a_value_to_a_closer_code_in_place():
    # Seven small values and one outlier. Once the fit moves the line,
    # 0.02 sits closer to code 6 than to its code 5.
    values = [0.02, -0.09, 0.10, 0.03, -0.04, 0.13, 0.04, -0.90]
    quantized = asymmetric.quantize(values, bits=4, block=8)
    assert list(quantized.unpacked_codes) == [5, 4, 7, 6, 5, 7, 6, -8]
    assert learned.alternate(quantized, values) is True
    assert list(quantized.unpacked_codes) == [6, 4, 7, 6, 5, 7, 6, -8]


def test_alternate_returns_false_until_the_codes_settle():
    # One scale for 4096 values, most near zero and a few far out. Its codes
    # keep moving for 150 passes, so the first call stops at 100.
    values = (-np.log((np.arange(4096) + 0.5) / 4096)).astype(np.float32)
    quantized = quantize_tensor(values, bits=6)
    assert learned.alternate(quantized, values) is False
    assert learned.alternate(quantized, values) is True


def squared_error(quantized, weights):
    return np.sum((quantized.dequantize() - weights) ** 2)


def test_alternate_ends_no_higher_than_refine_and_keeps_the_tensor():
    # Each block is wider than the last, so adaptive packs them at 4 to 7 bits.
    weights = (np.sin(np.arange(256)) * (1 + np.arange(256) // 32)).astype(np.float32)
    weights = weights.reshape(8, 32)
    for original in [
        quantize(weights, bits=4),
        asymmetric.quantize(weights, bits=4),
        adaptive.quantize(weights, block=32, tolerance=0.1),
    ]:
        refined = learned.refine(original.copy(), weights)
        alternated = original.copy()
        assert learned.alternate(alternated, weights)
        assert squared_error(alternated, weights) <= squared_error(refined, weights)
        assert alternated.kind == original.kind
        assert alternated.shape == (8, 32)
        assert alternated.nbytes == original.nbytes
    with pytest.raises(LengthMismatchError):
        learned.alternate(quantize(weights), weights[:4])


def test_fit_rejects_packed_codes():
    weights = [0.42, -0.10, 0.70, -0.50]
    quantized = quantize(weights, bits=8, block=4)
    with pytest.raises(TypeError, match="unpacked_codes"):
        learned.fit_scale_and_zero_point(weights, quantized.codes)
    scale, zero_point = learned.fit_scale_and_zero_point(
        weights, quantized.unpacked_codes
    )
    assert isinstance(scale, float)
    assert isinstance(zero_point, float)


def test_except_quantize_error_catches_length():
    with pytest.raises(QuantizeError, match="codes must have length 1, got 2") as raised:
        learned.fit_scale_and_zero_point([0.1], [0, 1])
    assert isinstance(raised.value, ValueError)


def test_a_copy_keeps_the_original_through_refine_and_alternate():
    weights = np.linspace(-0.5, 0.5, 64, dtype=np.float32).reshape(2, 32)
    for refit in [learned.refine, learned.alternate]:
        quantized = quantize(weights, bits=4)
        original = quantized.copy()
        refit(quantized, weights)
        assert quantized != original
        assert original == quantize(weights, bits=4)


def test_refine_and_alternate_work_while_other_threads_use_the_tensor():
    # Too large for any call to keep the GIL, so with a GIL too, the reads
    # can run while a refit does.
    weights = np.random.default_rng(0).standard_normal((512, 256)).astype(np.float32)
    quantized = quantize(weights, bits=4)
    versions = {quantized.to_bytes()}
    saved, products = set(), set()
    stop = threading.Event()

    def read_until_stopped():
        out = np.empty_like(weights)
        while not stop.is_set():
            products.add(quantized.matmul(weights[0]).tobytes())
            saved.add(quantized.to_bytes())
            quantized.dot(weights)
            quantized.dequantize(out)

    with ThreadPoolExecutor(4) as pool:
        readers = [pool.submit(read_until_stopped) for _ in range(4)]
        try:
            # Refitting to the weights and to their negation in turn changes
            # every scale each time.
            for target in [weights, -weights] * 20:
                learned.refine(quantized, target)
                versions.add(quantized.to_bytes())
                learned.alternate(quantized, target)
                versions.add(quantized.to_bytes())
        finally:
            stop.set()
        for reader in readers:
            reader.result()
    # Every read saw one whole version, even when a refit was stored mid-read.
    assert saved <= versions
    assert products <= {Quantized(version).matmul(weights[0]).tobytes() for version in versions}


def test_an_array_that_another_call_is_writing_gets_an_error_that_says_so():
    weights = np.random.default_rng(0).standard_normal((2048, 2048)).astype(np.float32)
    quantized = quantize(weights, bits=4)
    out = np.empty_like(weights)
    first_row = quantize(weights[:1], bits=4)
    in_use = "out is in use by another call, like one on another thread; give each call its own out"
    being_written = (
        "values is being written by another call, like a matmul or dequantize with out= "
        "on another thread; read it once that call returns"
    )
    messages = set()
    started, stop = threading.Event(), threading.Event()

    def keep_writing():
        started.set()
        while not stop.is_set():
            try:
                quantized.dequantize(out)
            except ValueError as error:
                # Without a GIL, a read below can start first.
                messages.add(str(error))

    with ThreadPoolExecutor(1) as pool:
        # Starting a thread can take longer than a write.
        pool.submit(lambda: None).result()
        writing = pool.submit(keep_writing)
        started.wait()
        deadline = time.perf_counter() + 10
        try:
            while messages != {in_use, being_written} and time.perf_counter() < deadline:
                try:
                    # This keeps the GIL, so no write starts while it reads.
                    learned.fit_scale_and_zero_point(out[0], first_row.unpacked_codes)
                    continue
                except ValueError as error:
                    messages.add(str(error))
                # A write is running, so this fails too.
                try:
                    first_row.dequantize(out[:1])
                except ValueError as error:
                    messages.add(str(error))
        finally:
            stop.set()
        writing.result()
    assert messages == {in_use, being_written}


@pytest.mark.skipif(
    sysconfig.get_config_var("Py_GIL_DISABLED") and not sys._is_gil_enabled(),
    reason="the GIL is off, so alternate can't hold it",
)
def test_other_threads_keep_running_while_alternate_refits():
    # A busy machine can pause a thread for a tenth of a second or more, so
    # the matrix is large enough that the refit takes several times that.
    weights = np.random.default_rng(0).standard_normal((2048, 2048)).astype(np.float32)
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
    # While the GIL is free, this loop goes around every microsecond or so,
    # but if alternate held the GIL, the loop would stop for almost the whole
    # refit. Half the refit falls between the two on fast and slow machines
    # alike. The loop doesn't sleep, since a 1 ms sleep can last 10 ms on
    # some machines.
    assert longest_pause < (last_check - start) / 2


@pytest.mark.skipif(
    sysconfig.get_config_var("Py_GIL_DISABLED") and not sys._is_gil_enabled(),
    reason="the GIL is off, so there's no GIL to keep",
)
def test_calls_on_at_most_65536_values_keep_the_gil():
    weights = np.random.default_rng(0).standard_normal((256, 256)).astype(np.float32)
    quantized = quantize(weights, bits=4)
    out = np.empty_like(weights)
    # alternate can go through its values 100 times, so it keeps the GIL
    # only for a hundredth as many.
    few = weights[:2]
    calls = [
        lambda: quantize(weights, bits=4),
        lambda: quantized.dequantize(),
        lambda: quantized.dequantize(out),
        lambda: quantized.dot(weights),
        lambda: quantized.matmul(weights[0], out=out[0]),
        lambda: learned.refine(quantized, weights),
        lambda: learned.alternate(quantize(few, bits=4), few),
    ]
    turns = 0
    go, stop = threading.Event(), threading.Event()

    def take_turns():
        nonlocal turns
        go.wait()
        while not stop.is_set():
            turns += 1

    switch_interval = sys.getswitchinterval()
    # The other thread waits this long before it asks for the GIL, longer
    # than the calls take, so it only takes a turn if a call gives the GIL up.
    sys.setswitchinterval(1.0)
    other = threading.Thread(target=take_turns)
    other.start()
    try:
        go.set()
        for call in calls:
            before = turns
            call()
            assert turns == before
    finally:
        stop.set()
        other.join()
        sys.setswitchinterval(switch_interval)


@pytest.mark.skipif(
    not sysconfig.get_config_var("Py_GIL_DISABLED"), reason="needs free-threaded Python"
)
def test_free_threaded_python_keeps_the_gil_off():
    # If importing turned the GIL back on, the thread tests above would pass
    # without running in parallel.
    assert not sys._is_gil_enabled()
