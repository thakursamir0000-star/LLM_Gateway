import pytest
import yaml
from pydantic import ValidationError

from gateway.config import DEFAULT_CONFIG, ROOT, load_config, resolve_path
from tests.conftest import make_config


def test_shipped_config_is_valid():
    config = load_config(DEFAULT_CONFIG)
    assert config.deployment.active in config.releases
    release = config.releases[config.deployment.active]
    assert config.models[release.cheap].model == "openai/gpt-oss-20b"
    assert config.models[release.strong].model == "openai/gpt-oss-120b"
    assert {"groq", "gemini"} <= set(config.providers)
    assert config.circuit_breaker.failure_threshold == 5
    assert config.circuit_breaker.recovery_timeout_s == 30.0
    assert config.circuit_breaker.half_open_probes == 1


def test_custom_circuit_breaker_values_override_defaults(tmp_path):
    data = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    data["circuit_breaker"] = {
        "failure_threshold": 2,
        "recovery_timeout_s": 7.5,
        "half_open_probes": 3,
    }
    path = tmp_path / "gateway.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    config = load_config(path)
    assert config.circuit_breaker.failure_threshold == 2
    assert config.circuit_breaker.recovery_timeout_s == 7.5
    assert config.circuit_breaker.half_open_probes == 3


def test_resolve_model_by_alias_or_provider_name(config):
    assert config.resolve_model("small") == "small"
    assert config.resolve_model("openai/gpt-oss-120b") == "large"
    assert config.resolve_model("gpt-5") is None


def test_cost_uses_prices_per_million(config):
    assert config.models["small"].cost(1_000_000, 0) == pytest.approx(0.10)
    assert config.models["small"].cost(2000, 1000) == pytest.approx((2000 * 0.10 + 1000 * 0.50) / 1e6)
    assert config.models["gemma"].cost(5000, 5000) == 0.0


@pytest.mark.parametrize("overrides, message", [
    ({"models": {"x": {"provider": "openai", "model": "gpt"}}}, "unknown provider"),
    ({"releases": {"v1": {"cheap": "nope", "strong": "large", "prompt": "plain@1"}}}, "unknown cheap model"),
    ({"releases": {"v1": {"cheap": "small", "strong": "large", "prompt": "missing@9"}}}, "unknown prompt"),
    ({"deployment": {"active": "v9"}}, "unknown release"),
    ({"deployment": {"active": "v1", "canary": {"release": "v1", "share": 0.1}}}, "must differ"),
    ({"deployment": {"active": "v1", "canary": {"release": "v2", "share": 1.5}}}, "less than or equal"),
    ({"routing": {"threshold": 2}}, "less than or equal"),
    ({"routing": {"classifier_threshold": 2}}, "less than or equal"),
    ({"routing": {"default_mode": "fastest"}}, "default_mode"),
])
def test_invalid_config_is_rejected(overrides, message):
    with pytest.raises(ValidationError, match=message):
        make_config(**overrides)


def test_unknown_keys_are_rejected():
    with pytest.raises(ValidationError, match="Extra inputs"):
        make_config(routing={"treshold": 0.4})


def test_env_overrides_db_path(monkeypatch, tmp_path):
    monkeypatch.setenv("GATEWAY_DB", str(tmp_path / "x.db"))
    assert load_config(DEFAULT_CONFIG).storage.db_path == str(tmp_path / "x.db")


def test_relative_paths_resolve_from_project_root(tmp_path):
    assert resolve_path("models/router.json") == ROOT / "models" / "router.json"
    assert resolve_path(tmp_path) == tmp_path
