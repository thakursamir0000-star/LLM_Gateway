"""Redis-backed distributed state backend for multi-instance deployments."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import redis

from gateway.circuit import CircuitState, ProbeToken
from gateway.config import CircuitBreakerConfig, RedisConfig
from gateway.ratelimit import RateLimited, Reservation
from gateway.state.base import CircuitBreakerProtocol, RateLimiterProtocol, StateBackend

log = logging.getLogger("gateway.state.redis")

WINDOW_S = 60.0

# ── Lua Scripts ───────────────────────────────────────────────────────

LUA_RATELIMIT_WAIT_AND_RESERVE = """
local zset_key = KEYS[1]
local items_key = KEYS[2]
local rpm = tonumber(ARGV[1])
local tpm = tonumber(ARGV[2])
local tokens = tonumber(ARGV[3])
local reservation_id = ARGV[4]
local reserve_if_ready = ARGV[5] == "1"

local time_arr = redis.call('TIME')
local now = tonumber(time_arr[1]) + (tonumber(time_arr[2]) / 1000000)
local window_s = 60.0
local cutoff = now - window_s

-- 1. Expire old reservations
local expired_ids = redis.call('ZRANGEBYSCORE', zset_key, '-inf', cutoff)
if #expired_ids > 0 then
    redis.call('ZREMRANGEBYSCORE', zset_key, '-inf', cutoff)
    for _, id in ipairs(expired_ids) do
        redis.call('HDEL', items_key, id)
    end
end

-- 2. Fetch current window in chronological order
local current_ids = redis.call('ZRANGE', zset_key, 0, -1, 'WITHSCORES')
local count = #current_ids / 2
local wait = 0.0

-- Check RPM
if rpm > 0 and count >= rpm then
    local idx = (count - rpm) * 2 + 2
    local oldest_t = tonumber(current_ids[idx])
    local rpm_wait = oldest_t + window_s - now
    if rpm_wait > wait then
        wait = rpm_wait
    end
end

-- Check TPM
if tpm > 0 and count > 0 then
    local total_used = 0
    local reservations = {}
    for i = 1, #current_ids, 2 do
        local id = current_ids[i]
        local t = tonumber(current_ids[i + 1])
        local item_raw = redis.call('HGET', items_key, id)
        local tok = tonumber(item_raw) or 0
        total_used = total_used + tok
        table.insert(reservations, {t = t, tokens = tok})
    end

    local excess = total_used + math.min(tokens, tpm) - tpm
    if excess > 0 then
        local freed = 0
        for _, r in ipairs(reservations) do
            freed = freed + r.tokens
            if freed >= excess then
                local tpm_wait = r.t + window_s - now
                if tpm_wait > wait then
                    wait = tpm_wait
                end
                break
            end
        end
    end
end

if wait < 0 then
    wait = 0.0
end

if wait == 0.0 and reserve_if_ready and reservation_id ~= "" then
    redis.call('ZADD', zset_key, now, reservation_id)
    redis.call('HSET', items_key, reservation_id, tostring(math.max(0, tokens)))
    redis.call('EXPIRE', zset_key, 120)
    redis.call('EXPIRE', items_key, 120)
end

return {tostring(wait), tostring(now)}
"""

LUA_RATELIMIT_SETTLE = """
local items_key = KEYS[1]
local reservation_id = ARGV[1]
local tokens = tonumber(ARGV[2])

if redis.call('HEXISTS', items_key, reservation_id) == 1 then
    redis.call('HSET', items_key, reservation_id, tostring(math.max(0, tokens)))
    return 1
else
    return 0
end
"""

LUA_COOLDOWN_SET = """
local key = KEYS[1]
local duration = tonumber(ARGV[1])

local time_arr = redis.call('TIME')
local now = tonumber(time_arr[1]) + (tonumber(time_arr[2]) / 1000000)
local new_expiry = now + duration

local current_expiry = redis.call('GET', key)
if not current_expiry or tonumber(current_expiry) < new_expiry then
    redis.call('SET', key, tostring(new_expiry), 'EX', math.ceil(duration))
    return tostring(new_expiry)
else
    return tostring(current_expiry)
end
"""

LUA_COOLDOWN_LEFT = """
local key = KEYS[1]
local time_arr = redis.call('TIME')
local now = tonumber(time_arr[1]) + (tonumber(time_arr[2]) / 1000000)

local current_expiry = redis.call('GET', key)
if not current_expiry then
    return "0"
end
local left = tonumber(current_expiry) - now
if left <= 0 then
    redis.call('DEL', key)
    return "0"
end
return tostring(left)
"""

LUA_CIRCUIT_CAN_EXECUTE = """
local hash_key = KEYS[1]
local probes_key = KEYS[2]
local failure_threshold = tonumber(ARGV[1])
local recovery_timeout = tonumber(ARGV[2])
local half_open_probes = tonumber(ARGV[3])
local probe_id = ARGV[4]

local time_arr = redis.call('TIME')
local now = tonumber(time_arr[1]) + (tonumber(time_arr[2]) / 1000000)

local state = redis.call('HGET', hash_key, 'state') or 'CLOSED'
local opened_at = tonumber(redis.call('HGET', hash_key, 'opened_at') or '0')

-- Expire stale probe leases (older than now)
redis.call('ZREMRANGEBYSCORE', probes_key, '-inf', now)

if state == 'OPEN' then
    if now - opened_at >= recovery_timeout then
        state = 'HALF_OPEN'
        redis.call('HSET', hash_key, 'state', 'HALF_OPEN', 'successes', 0)
        redis.call('DEL', probes_key)
    else
        return {0, 0, state}
    end
end

if state == 'HALF_OPEN' then
    local active_probes = redis.call('ZCARD', probes_key)
    if active_probes < half_open_probes then
        local lease_ttl = math.max(recovery_timeout, 30.0)
        redis.call('ZADD', probes_key, now + lease_ttl, probe_id)
        redis.call('EXPIRE', probes_key, math.ceil(lease_ttl * 2))
        redis.call('EXPIRE', hash_key, 86400)
        return {1, 1, state}
    else
        return {0, 0, state}
    end
end

return {1, 0, state}
"""

LUA_CIRCUIT_RECORD_SUCCESS = """
local hash_key = KEYS[1]
local probes_key = KEYS[2]
local probe_id = ARGV[1]
local half_open_probes = tonumber(ARGV[2])

if probe_id ~= "" then
    redis.call('ZREM', probes_key, probe_id)
    local state = redis.call('HGET', hash_key, 'state') or 'CLOSED'
    if state == 'HALF_OPEN' then
        local successes = redis.call('HINCRBY', hash_key, 'successes', 1)
        if successes >= half_open_probes then
            redis.call('HSET', hash_key, 'state', 'CLOSED', 'failures', 0, 'successes', 0)
            redis.call('DEL', probes_key)
        end
    end
else
    redis.call('HSET', hash_key, 'failures', 0)
end
return 1
"""

LUA_CIRCUIT_RECORD_FAILURE = """
local hash_key = KEYS[1]
local probes_key = KEYS[2]
local probe_id = ARGV[1]
local is_qualifying = ARGV[2] == "1"
local failure_threshold = tonumber(ARGV[3])

local time_arr = redis.call('TIME')
local now = tonumber(time_arr[1]) + (tonumber(time_arr[2]) / 1000000)

if probe_id ~= "" then
    redis.call('ZREM', probes_key, probe_id)
end

if not is_qualifying then
    return 0
end

if probe_id ~= "" then
    redis.call('HSET', hash_key, 'state', 'OPEN', 'opened_at', tostring(now), 'successes', 0)
    redis.call('DEL', probes_key)
else
    local failures = redis.call('HINCRBY', hash_key, 'failures', 1)
    if failures >= failure_threshold then
        redis.call('HSET', hash_key, 'state', 'OPEN', 'opened_at', tostring(now), 'successes', 0)
        redis.call('DEL', probes_key)
    end
end
return 1
"""

LUA_CIRCUIT_RECORD_NEUTRAL = """
local probes_key = KEYS[1]
local probe_id = ARGV[1]
if probe_id ~= "" then
    redis.call('ZREM', probes_key, probe_id)
end
return 1
"""

LUA_CIRCUIT_GET_STATE = """
local hash_key = KEYS[1]
local probes_key = KEYS[2]
local recovery_timeout = tonumber(ARGV[1])

local time_arr = redis.call('TIME')
local now = tonumber(time_arr[1]) + (tonumber(time_arr[2]) / 1000000)

local state = redis.call('HGET', hash_key, 'state') or 'CLOSED'
local opened_at = tonumber(redis.call('HGET', hash_key, 'opened_at') or '0')

redis.call('ZREMRANGEBYSCORE', probes_key, '-inf', now)
local active_probes = redis.call('ZCARD', probes_key)

if state == 'OPEN' and (now - opened_at >= recovery_timeout) then
    state = 'HALF_OPEN'
    redis.call('HSET', hash_key, 'state', 'HALF_OPEN', 'successes', 0)
    redis.call('DEL', probes_key)
    active_probes = 0
end

local failures = tonumber(redis.call('HGET', hash_key, 'failures') or '0')
local successes = tonumber(redis.call('HGET', hash_key, 'successes') or '0')

return {state, tostring(failures), tostring(active_probes), tostring(successes)}
"""

LUA_CIRCUIT_RESET = """
local hash_key = KEYS[1]
local probes_key = KEYS[2]
redis.call('HSET', hash_key, 'state', 'CLOSED', 'failures', 0, 'successes', 0, 'opened_at', '0')
redis.call('DEL', probes_key)
return 1
"""


class RedisCooldowns:
    def __init__(self, backend: RedisStateBackend):
        self._backend = backend

    def cooldown_left(self, alias: str) -> float:
        try:
            key = f"gateway:cooldown:{alias}"
            res = self._backend.scripts["cooldown_left"](keys=[key])
            return max(0.0, float(res))
        except (redis.ConnectionError, redis.TimeoutError) as e:
            self._backend.mark_degraded(e)
            return self._backend.fallback_backend.cooldown_left(alias)

    def set_cooldown(self, alias: str, duration_s: float) -> None:
        try:
            key = f"gateway:cooldown:{alias}"
            self._backend.scripts["cooldown_set"](keys=[key], args=[duration_s])
        except (redis.ConnectionError, redis.TimeoutError) as e:
            self._backend.mark_degraded(e)
            self._backend.fallback_backend.set_cooldown(alias, duration_s)


class RedisRateLimiter(RateLimiterProtocol):
    def __init__(self, alias: str, rpm: int, tpm: int, backend: RedisStateBackend):
        self.alias = alias
        self.rpm = rpm
        self.tpm = tpm
        self._backend = backend
        self._zset_key = f"gateway:ratelimit:{alias}:z"
        self._items_key = f"gateway:ratelimit:{alias}:items"

    def wait_time(self, tokens: int) -> float:
        try:
            res = self._backend.scripts["ratelimit_wait_and_reserve"](
                keys=[self._zset_key, self._items_key],
                args=[self.rpm, self.tpm, tokens, "", "0"],
            )
            return float(res[0])
        except (redis.ConnectionError, redis.TimeoutError) as e:
            self._backend.mark_degraded(e)
            fb = self._backend.fallback_backend.get_rate_limiter(self.alias, self.rpm, self.tpm)
            return fb.wait_time(tokens)

    def reserve(self, tokens: int) -> Reservation:
        reservation_id = uuid.uuid4().hex
        try:
            res = self._backend.scripts["ratelimit_wait_and_reserve"](
                keys=[self._zset_key, self._items_key],
                args=[self.rpm, self.tpm, tokens, reservation_id, "1"],
            )
            t = float(res[1])
            return Reservation(t=t, tokens=tokens, model=self.alias, reservation_id=reservation_id)
        except (redis.ConnectionError, redis.TimeoutError) as e:
            self._backend.mark_degraded(e)
            fb = self._backend.fallback_backend.get_rate_limiter(self.alias, self.rpm, self.tpm)
            return fb.reserve(tokens)

    def settle(self, reservation: Reservation, tokens: int) -> None:
        reservation.tokens = max(0, tokens)
        if not reservation.reservation_id:
            return
        try:
            self._backend.scripts["ratelimit_settle"](
                keys=[self._items_key],
                args=[reservation.reservation_id, tokens],
            )
        except (redis.ConnectionError, redis.TimeoutError) as e:
            self._backend.mark_degraded(e)
            fb = self._backend.fallback_backend.get_rate_limiter(self.alias, self.rpm, self.tpm)
            fb.settle(reservation, tokens)

    async def acquire(
        self,
        tokens: int,
        max_wait: float = float("inf"),
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> Reservation:
        reservation_id = uuid.uuid4().hex
        while True:
            try:
                res = self._backend.scripts["ratelimit_wait_and_reserve"](
                    keys=[self._zset_key, self._items_key],
                    args=[self.rpm, self.tpm, tokens, reservation_id, "1"],
                )
                wait = float(res[0])
                if wait <= 0:
                    t = float(res[1])
                    return Reservation(t=t, tokens=tokens, model=self.alias, reservation_id=reservation_id)
                if wait > max_wait:
                    raise RateLimited(wait)
            except (redis.ConnectionError, redis.TimeoutError) as e:
                self._backend.mark_degraded(e)
                fb = self._backend.fallback_backend.get_rate_limiter(self.alias, self.rpm, self.tpm)
                return await fb.acquire(tokens, max_wait=max_wait, sleep=sleep)
            await sleep(wait + 0.05)


class RedisCircuitBreaker(CircuitBreakerProtocol):
    def __init__(self, name: str, config: CircuitBreakerConfig, backend: RedisStateBackend):
        self.name = name
        self.config = config
        self._backend = backend
        self._hash_key = f"gateway:circuit:{name}"
        self._probes_key = f"gateway:circuit:{name}:probes"

    @property
    def state(self) -> CircuitState:
        try:
            res = self._backend.scripts["circuit_get_state"](
                keys=[self._hash_key, self._probes_key],
                args=[self.config.recovery_timeout_s],
            )
            state_str = res[0]
            if isinstance(state_str, bytes):
                state_str = state_str.decode("utf-8")
            return CircuitState(state_str)
        except (redis.ConnectionError, redis.TimeoutError) as e:
            self._backend.mark_degraded(e)
            return self._backend.fallback_backend.get_circuit_breaker(self.name, self.config).state

    @property
    def consecutive_failures(self) -> int:
        try:
            res = self._backend.scripts["circuit_get_state"](
                keys=[self._hash_key, self._probes_key],
                args=[self.config.recovery_timeout_s],
            )
            return int(res[1])
        except (redis.ConnectionError, redis.TimeoutError) as e:
            self._backend.mark_degraded(e)
            return self._backend.fallback_backend.get_circuit_breaker(self.name, self.config).consecutive_failures

    @property
    def active_probes(self) -> int:
        try:
            res = self._backend.scripts["circuit_get_state"](
                keys=[self._hash_key, self._probes_key],
                args=[self.config.recovery_timeout_s],
            )
            return int(res[2])
        except (redis.ConnectionError, redis.TimeoutError) as e:
            self._backend.mark_degraded(e)
            return self._backend.fallback_backend.get_circuit_breaker(self.name, self.config).active_probes

    def can_execute(self) -> tuple[bool, ProbeToken]:
        probe_id = uuid.uuid4().hex
        try:
            res = self._backend.scripts["circuit_can_execute"](
                keys=[self._hash_key, self._probes_key],
                args=[
                    self.config.failure_threshold,
                    self.config.recovery_timeout_s,
                    self.config.half_open_probes,
                    probe_id,
                ],
            )
            allowed = bool(int(res[0]))
            is_probe = bool(int(res[1]))
            return allowed, ProbeToken(
                allowed=allowed,
                is_probe=is_probe,
                provider=self.name,
                token_id=probe_id if is_probe else "",
            )
        except (redis.ConnectionError, redis.TimeoutError) as e:
            self._backend.mark_degraded(e)
            return self._backend.fallback_backend.get_circuit_breaker(self.name, self.config).can_execute()

    def record_success(self, token: ProbeToken) -> None:
        try:
            self._backend.scripts["circuit_record_success"](
                keys=[self._hash_key, self._probes_key],
                args=[token.token_id, self.config.half_open_probes],
            )
        except (redis.ConnectionError, redis.TimeoutError) as e:
            self._backend.mark_degraded(e)
            self._backend.fallback_backend.get_circuit_breaker(self.name, self.config).record_success(token)

    def record_failure(self, token: ProbeToken, is_qualifying: bool = True) -> None:
        try:
            self._backend.scripts["circuit_record_failure"](
                keys=[self._hash_key, self._probes_key],
                args=[token.token_id, "1" if is_qualifying else "0", self.config.failure_threshold],
            )
        except (redis.ConnectionError, redis.TimeoutError) as e:
            self._backend.mark_degraded(e)
            self._backend.fallback_backend.get_circuit_breaker(self.name, self.config).record_failure(
                token, is_qualifying=is_qualifying
            )

    def record_neutral(self, token: ProbeToken) -> None:
        try:
            self._backend.scripts["circuit_record_neutral"](
                keys=[self._probes_key],
                args=[token.token_id],
            )
        except (redis.ConnectionError, redis.TimeoutError) as e:
            self._backend.mark_degraded(e)
            self._backend.fallback_backend.get_circuit_breaker(self.name, self.config).record_neutral(token)

    def reset(self) -> None:
        try:
            self._backend.scripts["circuit_reset"](
                keys=[self._hash_key, self._probes_key],
                args=[],
            )
        except (redis.ConnectionError, redis.TimeoutError) as e:
            self._backend.mark_degraded(e)
            self._backend.fallback_backend.get_circuit_breaker(self.name, self.config).reset()


class RedisStateBackend(StateBackend):
    """Production Redis-backed shared state backend."""

    def __init__(self, config: RedisConfig):
        self.config = config
        self._pool = redis.ConnectionPool.from_url(
            config.url,
            socket_timeout=config.socket_timeout_s,
            socket_connect_timeout=config.connect_timeout_s,
            max_connections=config.max_connections,
            decode_responses=True,
        )
        self.client = redis.Redis(connection_pool=self._pool)
        self.cooldowns = RedisCooldowns(self)
        self._limiters: dict[str, RedisRateLimiter] = {}
        self._breakers: dict[str, RedisCircuitBreaker] = {}
        self.fallback_backend = None

        from gateway.state.memory import InMemoryStateBackend

        self.fallback_backend = InMemoryStateBackend()
        self._degraded = False
        self._last_error = ""

        # Register Lua scripts
        self.scripts = {
            "ratelimit_wait_and_reserve": self.client.register_script(LUA_RATELIMIT_WAIT_AND_RESERVE),
            "ratelimit_settle": self.client.register_script(LUA_RATELIMIT_SETTLE),
            "cooldown_set": self.client.register_script(LUA_COOLDOWN_SET),
            "cooldown_left": self.client.register_script(LUA_COOLDOWN_LEFT),
            "circuit_can_execute": self.client.register_script(LUA_CIRCUIT_CAN_EXECUTE),
            "circuit_record_success": self.client.register_script(LUA_CIRCUIT_RECORD_SUCCESS),
            "circuit_record_failure": self.client.register_script(LUA_CIRCUIT_RECORD_FAILURE),
            "circuit_record_neutral": self.client.register_script(LUA_CIRCUIT_RECORD_NEUTRAL),
            "circuit_get_state": self.client.register_script(LUA_CIRCUIT_GET_STATE),
            "circuit_reset": self.client.register_script(LUA_CIRCUIT_RESET),
        }

    def mark_degraded(self, error: Exception) -> None:
        if not self._degraded:
            log.warning("Redis state backend connection failure: %s (using local fallback)", error)
        self._degraded = True
        self._last_error = str(error)

    def cooldown_left(self, alias: str) -> float:
        return self.cooldowns.cooldown_left(alias)

    def set_cooldown(self, alias: str, duration_s: float) -> None:
        self.cooldowns.set_cooldown(alias, duration_s)

    def get_rate_limiter(self, alias: str, rpm: int, tpm: int) -> RateLimiterProtocol:
        key = f"{alias}:{rpm}:{tpm}"
        if key not in self._limiters:
            self._limiters[key] = RedisRateLimiter(alias=alias, rpm=rpm, tpm=tpm, backend=self)
        return self._limiters[key]

    def get_circuit_breaker(self, name: str, config: CircuitBreakerConfig) -> CircuitBreakerProtocol:
        if name not in self._breakers:
            self._breakers[name] = RedisCircuitBreaker(name=name, config=config, backend=self)
        return self._breakers[name]

    def is_ready(self) -> tuple[bool, str]:
        try:
            self.client.ping()
            self._degraded = False
            self._last_error = ""
            return True, "redis ready"
        except Exception as e:
            self.mark_degraded(e)
            return False, f"redis error: {e}"

    def status(self) -> dict[str, Any]:
        ready, msg = self.is_ready()
        return {
            "backend": "redis",
            "ready": ready,
            "distributed": True,
            "degraded": self._degraded,
            "last_error": self._last_error if self._degraded else None,
            "url": self.config.url.split("@")[-1],  # hide auth credentials
        }

    def close(self) -> None:
        try:
            self.client.close()
            self._pool.disconnect()
        except Exception:
            pass
