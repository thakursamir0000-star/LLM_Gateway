"""Hashed client keys, keys from a file, several keys per app (rotation), hashed admin key."""

from __future__ import annotations

import hashlib
import random

from fastapi.testclient import TestClient

import scripts.make_api_key as make_api_key
from gateway.app import create_app, key_matches, parse_api_keys
from gateway.monitoring.store import Store
from gateway.providers.fake import FakeProvider
from gateway.routing import RulesRouter
from tests.conftest import make_config

EASY = {"messages": [{"role": "user", "content": "Hello"}]}


def sha(key: str) -> str:
    return "sha256:" + hashlib.sha256(key.encode()).hexdigest()


def client_for(tmp_path, config=None):
    store = Store(tmp_path / "gateway.db")
    app = create_app(config or make_config(auth={"allow_anonymous": False}), store=store, router=RulesRouter(),
                     providers={"groq": FakeProvider("groq"), "gemini": FakeProvider("gemini")},
                     rng=random.Random(0))
    return TestClient(app), store


def chat(client, key):
    return client.post("/v1/chat/completions", json=EASY, headers={"Authorization": f"Bearer {key}"})


def test_parse_hashed_entries_files_and_comments():
    raw = f"# rotated 2026-11\nweb:{sha('k1').upper().replace('SHA256', 'sha256')}\nweb:k2, batch:k3"
    keys = parse_api_keys(raw)
    assert keys == {sha("k1"): "web", "k2": "web", "k3": "batch"}
    assert key_matches("k1", sha("k1")) and not key_matches("k2", sha("k1")) and key_matches("k2", "k2")


def test_hashed_keys_from_a_file_and_rotation(monkeypatch, tmp_path):
    old_key, old_entry = make_api_key.make_key("web")
    new_key, new_entry = make_api_key.make_key("web")
    keys_file = tmp_path / "api-keys"
    keys_file.write_text(f"{old_entry}\n{new_entry}\n", encoding="utf-8")
    assert old_key not in keys_file.read_text(encoding="utf-8")  # only hashes are stored
    monkeypatch.delenv("TEST_GATEWAY_KEYS", raising=False)
    monkeypatch.setenv("TEST_GATEWAY_KEYS_FILE", str(keys_file))

    client, store = client_for(tmp_path)
    with client:
        assert chat(client, old_key).status_code == 200  # both keys work while the client moves over
        assert chat(client, new_key).status_code == 200
    store.close()

    keys_file.write_text(f"{new_entry}\n", encoding="utf-8")  # the old key is revoked; replicas restart
    client, store = client_for(tmp_path)
    with client:
        assert chat(client, old_key).status_code == 401
        assert chat(client, new_key).status_code == 200
    store.close()


def test_hashed_admin_key(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_GATEWAY_KEYS", "web:k")
    monkeypatch.setenv("TEST_GATEWAY_ADMIN_KEY", sha("admin-secret"))
    client, store = client_for(tmp_path)
    with client:
        assert client.get("/admin/status", headers={"X-Admin-Key": "admin-secret"}).status_code == 200
        assert client.get("/admin/status", headers={"X-Admin-Key": sha("admin-secret")}).status_code == 401
    store.close()


def test_make_api_key_appends_only_the_hash(tmp_path, capsys):
    keys_file = tmp_path / "keys"
    assert make_api_key.main(["web", "--append", str(keys_file)]) == 0
    out = capsys.readouterr().out
    key = out.rsplit(": ", 1)[1].strip()
    assert keys_file.read_text(encoding="utf-8").strip() == f"web:{sha(key)}"
