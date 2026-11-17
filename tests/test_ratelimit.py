import asyncio

import pytest

from gateway.ratelimit import RateLimited, RateLimiter, estimate_tokens


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_no_limits_never_wait():
    limiter = RateLimiter()
    for _ in range(100):
        assert limiter.wait_time(10_000) == 0
        limiter.reserve(10_000)


def test_rpm_waits_for_the_oldest_request_to_leave_the_window():
    clock = Clock()
    limiter = RateLimiter(rpm=2, clock=clock)
    limiter.reserve(1)
    clock.t += 10
    limiter.reserve(1)
    assert limiter.wait_time(1) == pytest.approx(50)
    clock.t += 50
    assert limiter.wait_time(1) == 0


def test_tpm_waits_until_enough_tokens_expire():
    clock = Clock()
    limiter = RateLimiter(tpm=1000, clock=clock)
    limiter.reserve(600)
    clock.t += 20
    limiter.reserve(300)
    assert limiter.wait_time(100) == 0  # 600 + 300 + 100 fits
    assert limiter.wait_time(200) == pytest.approx(40)  # needs the first 600 to expire
    clock.t += 40
    assert limiter.wait_time(200) == 0


def test_request_bigger_than_budget_runs_alone():
    clock = Clock()
    limiter = RateLimiter(tpm=1000, clock=clock)
    assert limiter.wait_time(5000) == 0
    limiter.reserve(5000)
    assert limiter.wait_time(5000) == pytest.approx(60)


def test_settle_replaces_the_estimate():
    clock = Clock()
    limiter = RateLimiter(tpm=1000, clock=clock)
    reservation = limiter.reserve(900)
    assert limiter.wait_time(500) > 0
    limiter.settle(reservation, 100)
    assert limiter.wait_time(500) == 0


def test_acquire_sleeps_then_reserves():
    clock = Clock()
    limiter = RateLimiter(rpm=1, clock=clock)
    limiter.reserve(1)
    waits = []

    async def sleep(seconds):
        waits.append(seconds)
        clock.t += seconds

    asyncio.run(limiter.acquire(1, sleep=sleep))
    assert waits and waits[0] == pytest.approx(60, abs=0.1)


def test_acquire_refuses_long_waits():
    clock = Clock()
    limiter = RateLimiter(rpm=1, clock=clock)
    limiter.reserve(1)
    with pytest.raises(RateLimited) as info:
        asyncio.run(limiter.acquire(1, max_wait=5))
    assert info.value.wait_s == pytest.approx(60)


def test_estimate_tokens():
    assert estimate_tokens(400, None) == 100 + 512
    assert estimate_tokens(400, 50) == 150
