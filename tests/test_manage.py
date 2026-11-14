import random
from datetime import UTC, datetime

import pytest

from gateway.config import BudgetConfig
from gateway.manage import BudgetPolicy, ReleaseError, ReleaseManager
from gateway.manage.__main__ import main as cli
from gateway.manage.budgets import seconds_until_utc_midnight
from gateway.manage.releases import DeploymentState, sticky_bucket
from gateway.monitoring.store import Store
from tests.conftest import make_config


@pytest.fixture
def manager(config):
    return ReleaseManager(config, Store(":memory:"))


# ── releases ────────────────────────────────────────────────────────

def test_state_starts_from_the_config(manager):
    state = manager.state()
    assert (state.active, state.canary, state.source) == ("v1", None, "config")


def test_config_canary_is_the_initial_state():
    config = make_config(deployment={"active": "v1", "canary": {"release": "v2", "share": 0.25}})
    state = ReleaseManager(config, Store(":memory:")).state()
    assert (state.canary, state.canary_share) == ("v2", 0.25)


def test_activate_and_rollback(manager):
    manager.activate("v2", note="try gemma")
    assert manager.state().active == "v2" and manager.state().source == "log"
    assert manager.rollback().active == "v1"
    assert manager.rollback().active == "v2"  # undoing the rollback
    actions = [row["action"] for row in manager.history()]
    assert actions == ["rollback", "rollback", "activate"]
    assert manager.history()[-1]["note"] == "try gemma"


def test_nothing_to_roll_back_to(manager):
    with pytest.raises(ReleaseError, match="nothing to roll back"):
        manager.rollback()


@pytest.mark.parametrize("action", [
    lambda m: m.activate("v9"),
    lambda m: m.activate("v1"),  # already active
    lambda m: m.set_canary("v1", 0.1),  # the active release
    lambda m: m.set_canary("v2", 1.5),
    lambda m: m.clear_canary(),  # none running
])
def test_invalid_changes_are_refused(manager, action):
    with pytest.raises(ReleaseError):
        action(manager)


def test_canary_lifecycle(manager):
    state = manager.set_canary("v2", 0.1)
    assert (state.active, state.canary, state.canary_share) == ("v1", "v2", 0.1)
    promoted = manager.activate("v2")  # promoting the canary ends it
    assert (promoted.active, promoted.canary) == ("v2", None)
    manager.set_canary("v1", 0.5)
    rolled = manager.rollback()  # rollback stops the canary
    assert (rolled.active, rolled.canary) == ("v1", None)


def test_canary_traffic_share_and_stickiness():
    state = DeploymentState("v1", "v2", 0.2)
    rng = random.Random(42)
    picks = [ReleaseManager.pick(state, None, rng) for _ in range(5000)]
    share = sum(is_canary for _, is_canary in picks) / len(picks)
    assert 0.17 < share < 0.23
    assert all(release == ("v2" if is_canary else "v1") for release, is_canary in picks)
    user_picks = {ReleaseManager.pick(state, "user-7", random.Random(i))[1] for i in range(20)}
    assert len(user_picks) == 1  # a user always lands on the same side
    assert ReleaseManager.pick(DeploymentState("v1"), "u", rng) == ("v1", False)
    assert 0 <= sticky_bucket("x") < 1


def test_log_naming_an_unknown_release_falls_back_to_config(config):
    store = Store(":memory:")
    store.add_deployment("activate", "deleted-release", None, 0, "", "2026-11-15T00:00:00+00:00")
    assert ReleaseManager(config, store).state().active == "v1"


def test_cli(tmp_path, monkeypatch, capsys):
    import yaml

    from tests.conftest import BASE_CONFIG

    path = tmp_path / "gateway.yaml"
    path.write_text(yaml.safe_dump({**BASE_CONFIG, "storage": {"db_path": str(tmp_path / "gw.db")}}))
    assert cli(["--config", str(path), "status"]) == 0
    assert "active:  v1" in capsys.readouterr().out
    assert cli(["--config", str(path), "canary", "v2", "0.1"]) == 0
    assert "v2 on 10% of traffic" in capsys.readouterr().out
    assert cli(["--config", str(path), "activate", "v2", "--note", "ship"]) == 0
    assert cli(["--config", str(path), "rollback"]) == 0
    assert "active:  v1" in capsys.readouterr().out
    assert cli(["--config", str(path), "history"]) == 0
    assert "# ship" in capsys.readouterr().out
    assert cli(["--config", str(path), "activate", "nope"]) == 1
    assert "unknown release" in capsys.readouterr().err


# ── budgets ─────────────────────────────────────────────────────────

def test_budget_limits_per_app():
    policy = BudgetPolicy(BudgetConfig(default_daily_usd=1.0, soft_limit_fraction=0.8,
                                       apps={"big": {"daily_usd": None}, "small": {"daily_usd": 0.1}}))
    assert policy.limit("anyone") == 1.0 and policy.limit("big") is None and policy.limit("small") == 0.1
    fine = policy.status("anyone", 0.5)
    assert not fine.soft and not fine.exceeded and fine.remaining == pytest.approx(0.5)
    assert policy.status("anyone", 0.85).soft and not policy.status("anyone", 0.85).exceeded
    assert policy.status("small", 0.1).exceeded
    unlimited = policy.status("big", 1e6)
    assert not unlimited.exceeded and unlimited.remaining is None


def test_no_default_budget_means_unlimited():
    assert not BudgetPolicy(BudgetConfig()).status("x", 1e9).exceeded


def test_seconds_until_utc_midnight():
    assert seconds_until_utc_midnight(datetime(2026, 11, 15, 23, 0, tzinfo=UTC)) == 3600
