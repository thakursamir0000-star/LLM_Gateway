"""Releases: which models and prompt version serve traffic, with rollback and canary.

Releases are defined in the config. The deployment state (the active release
and an optional canary with its traffic share) starts from ``deployment`` in
the config; every change is appended to the ``deployments`` table, and the
newest row is the current state. Rollback re-activates the release that was
active before the current one (like ``kubectl rollout undo``, two rollbacks
return to where you started) and stops any canary.
"""

from __future__ import annotations

import hashlib
import logging
import random
from dataclasses import dataclass

from gateway.config import GatewayConfig
from gateway.monitoring.store import Store, iso, utc_now

log = logging.getLogger("gateway.releases")


class ReleaseError(ValueError):
    """An invalid deployment change (unknown release, nothing to roll back to...)."""


@dataclass(frozen=True)
class DeploymentState:
    active: str
    canary: str | None = None
    canary_share: float = 0.0
    source: str = "config"  # config | log
    ts: str | None = None


def sticky_bucket(key: str) -> float:
    """A stable number in [0, 1) for ``key``, so a user stays on one side of a canary."""
    return int(hashlib.sha256(key.encode()).hexdigest()[:12], 16) / 16**12


class ReleaseManager:
    def __init__(self, config: GatewayConfig, store: Store):
        self.config = config
        self.store = store

    def _initial(self) -> DeploymentState:
        canary = self.config.deployment.canary
        return DeploymentState(self.config.deployment.active, canary.release if canary else None,
                               canary.share if canary else 0.0)

    def _valid(self, state: DeploymentState) -> bool:
        return state.active in self.config.releases and (state.canary is None or state.canary in self.config.releases)

    def state(self) -> DeploymentState:
        rows = self.store.deployments(limit=1)
        if not rows:
            return self._initial()
        row = rows[0]
        state = DeploymentState(row["active"], row["canary_release"], row["canary_share"] or 0.0, "log", row["ts"])
        if not self._valid(state):
            log.warning("deployment log names a release missing from the config (%s); using the config", state)
            return self._initial()
        return state

    def history(self, limit: int = 20) -> list[dict]:
        return self.store.deployments(limit)

    def _record(self, action: str, state: DeploymentState, note: str) -> DeploymentState:
        ts = iso(utc_now())
        self.store.add_deployment(action, state.active, state.canary, state.canary_share, note, ts)
        log.info("deployment %s: active=%s canary=%s share=%.2f", action, state.active, state.canary,
                 state.canary_share)
        return DeploymentState(state.active, state.canary, state.canary_share, "log", ts)

    def _check_release(self, release: str) -> None:
        if release not in self.config.releases:
            raise ReleaseError(f"unknown release {release!r} (known: {', '.join(self.config.releases)})")

    def activate(self, release: str, note: str = "") -> DeploymentState:
        """Make ``release`` the active one. Promoting the canary ends the canary."""
        self._check_release(release)
        current = self.state()
        if release == current.active:
            raise ReleaseError(f"{release!r} is already active")
        keep_canary = current.canary if current.canary != release else None
        return self._record("activate", DeploymentState(release, keep_canary,
                                                        current.canary_share if keep_canary else 0.0), note)

    def rollback(self, note: str = "") -> DeploymentState:
        """Re-activate the release that was active before the current one; stops any canary."""
        current = self.state()
        previous = None
        for row in self.store.deployments(limit=1000)[1:]:
            if row["active"] != current.active and row["active"] in self.config.releases:
                previous = row["active"]
                break
        if previous is None and self._initial().active != current.active:
            previous = self._initial().active
        if previous is None:
            raise ReleaseError("nothing to roll back to: no earlier active release")
        return self._record("rollback", DeploymentState(previous), note)

    def set_canary(self, release: str, share: float, note: str = "") -> DeploymentState:
        self._check_release(release)
        if not 0.0 <= share <= 1.0:
            raise ReleaseError("canary share must be between 0 and 1")
        current = self.state()
        if release == current.active:
            raise ReleaseError(f"{release!r} is the active release; a canary must be a different release")
        return self._record("canary", DeploymentState(current.active, release, share), note)

    def clear_canary(self, note: str = "") -> DeploymentState:
        current = self.state()
        if current.canary is None:
            raise ReleaseError("no canary is running")
        return self._record("canary-off", DeploymentState(current.active), note)

    @staticmethod
    def pick(state: DeploymentState, sticky_key: str | None, rng: random.Random) -> tuple[str, bool]:
        """(release, is_canary) for one request; ``sticky_key`` (e.g. the user id) keeps a user on one side."""
        if state.canary and state.canary_share > 0:
            draw = sticky_bucket(sticky_key) if sticky_key else rng.random()
            if draw < state.canary_share:
                return state.canary, True
        return state.active, False
