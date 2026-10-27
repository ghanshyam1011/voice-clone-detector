"""A streaming call must be assessed on its NEWEST audio. These tests pin the failure that made the
verdict fall behind real time: when feeding and assessing shared one loop, each (slow) assessment
consumed only half a second of audio, so the lag grew by the difference every second."""

import asyncio
import struct

import numpy as np
import pytest

from voiceguard.serve.session import LIVE_HOP_S, LiveBuffer

SR = 16000


def _f32(n, value=0.1):
    return struct.pack(f"<{n}f", *([value] * n))


def test_becomes_ready_after_half_a_second_of_new_audio_and_snapshot_resets_the_count():
    lb = LiveBuffer(SR)
    assert not lb.feed(_f32(int(SR * 0.4)))
    assert lb.snapshot() is None  # under half a second in total: nothing to assess yet
    assert lb.feed(_f32(int(SR * 0.2)))
    audio, heard = lb.snapshot()
    assert len(audio) == int(SR * 0.6) and heard == pytest.approx(0.6)
    assert not lb.feed(_f32(int(SR * 0.3)))  # a snapshot marked what came before as assessed
    assert lb.feed(_f32(int(SR * 0.3)))


def test_keeps_only_the_newest_seconds_but_the_call_clock_keeps_counting():
    lb = LiveBuffer(SR, cap_s=2.0)
    for _ in range(5):
        lb.feed(_f32(SR))
    audio, heard = lb.snapshot()
    assert len(audio) == 2 * SR and heard == pytest.approx(5.0)


def test_audio_at_another_rate_is_resampled_and_counted_in_call_seconds():
    lb = LiveBuffer(SR)
    lb.set_rate(48000)
    lb.feed(_f32(48000, 0.2))
    audio, heard = lb.snapshot()
    assert abs(len(audio) - SR) <= 2 and heard == pytest.approx(1.0, abs=0.01)


def test_a_chunk_boundary_inside_a_sample_does_not_misalign_what_follows():
    lb = LiveBuffer(SR)
    raw = _f32(SR)
    lb.feed(raw[:6])  # one and a half samples
    lb.feed(raw[6:])
    audio, _ = lb.snapshot()
    assert len(audio) == SR and np.allclose(audio, 0.1)


def test_not_a_number_becomes_silence_instead_of_reaching_the_models():
    lb = LiveBuffer(SR)
    lb.feed(struct.pack("<4f", float("nan"), 0.5, float("inf"), 0.25) + _f32(SR))
    audio, _ = lb.snapshot()
    assert np.isfinite(audio).all()


def test_a_slow_assessor_always_sees_the_newest_audio_and_never_falls_further_behind():
    """Audio arrives in real time (40 ms chunks); each assessment takes 0.8 s (measured on the
    dev laptop), longer than the half-second hop allows when feeding and assessing share one
    loop. The assessor must skip ahead, so what it sees is at most one assessment old and the
    final one covers the whole call."""
    seconds, assess_s, speed = 4.0, 0.8, 1.0  # real time: Windows timers can't fake a faster one
    seen = []

    async def run():
        lb = LiveBuffer(SR)
        wake, done = asyncio.Event(), asyncio.Event()
        chunk = int(SR * 0.04)

        async def feeder():
            for _ in range(int(seconds / 0.04)):
                if lb.feed(_f32(chunk)):
                    wake.set()
                await asyncio.sleep(0.04 / speed)
            done.set()
            wake.set()

        async def assessor():
            while True:
                await wake.wait()
                wake.clear()
                snap = lb.snapshot()
                if snap is not None:
                    await asyncio.sleep(assess_s / speed)  # the assessment itself
                    seen.append((snap[1], lb.heard / SR))  # (covered, arrived by then)
                if done.is_set() and lb.since == 0:
                    return
                if done.is_set():
                    wake.set()

        await asyncio.gather(feeder(), assessor())

    asyncio.run(run())
    covered_last, arrived_last = seen[-1]
    assert covered_last == pytest.approx(seconds, abs=0.05)  # the last one covers everything
    lag = max(arrived - covered for covered, arrived in seen)
    assert lag <= assess_s + LIVE_HOP_S + 0.1  # one assessment behind at most, never growing
    assert len(seen) < seconds / LIVE_HOP_S  # it skipped stale audio instead of queueing it
