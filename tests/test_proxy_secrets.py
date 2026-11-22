"""The dashboard password file for the HTTPS proxy (scripts/make_proxy_secrets.py)."""

from __future__ import annotations

import scripts.make_proxy_secrets as secrets_script


def test_apr1_matches_openssl():
    # Reference values from `openssl passwd -apr1 -salt <salt> <password>` (OpenSSL 3.5.7).
    assert secrets_script.apr1("password", "abcdefgh") == "$apr1$abcdefgh$FBwExRW4dCc8aL.OvjpIE1"
    assert secrets_script.apr1("correct horse battery staple", "12345678") == "$apr1$12345678$5s8zqQNNXgdW9osfkSNGf0"


def test_generated_password_is_saved_not_printed(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(secrets_script, "make_certificate", lambda directory, days: None)
    htpasswd = secrets_script.make_secrets(tmp_path, "ops")
    password = (tmp_path / "dashboard-password.txt").read_text(encoding="utf-8").strip()
    user, hashed = htpasswd.read_text(encoding="utf-8").strip().split(":", 1)
    assert user == "ops" and len(password) >= 20 and password not in hashed
    assert hashed == secrets_script.apr1(password, hashed.split("$")[2])  # the hash checks out for that password
    assert password not in capsys.readouterr().out


def test_given_password_writes_no_password_file(tmp_path, monkeypatch):
    monkeypatch.setattr(secrets_script, "make_certificate", lambda directory, days: None)
    htpasswd = secrets_script.make_secrets(tmp_path, "admin", "s3cret-passphrase")
    assert not (tmp_path / "dashboard-password.txt").exists()
    hashed = htpasswd.read_text(encoding="utf-8").strip().split(":", 1)[1]
    assert hashed == secrets_script.apr1("s3cret-passphrase", hashed.split("$")[2])
