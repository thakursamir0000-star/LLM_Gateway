"""Create the files the HTTPS proxy reads: a development TLS certificate and the dashboard password file.

    python scripts/make_proxy_secrets.py                       # into config/tls/ (git-ignored)
    DASHBOARD_PASSWORD=... python scripts/make_proxy_secrets.py --user admin --dir config/tls

Writes ``server.key`` and ``server.crt`` (self-signed for localhost and 127.0.0.1, valid ``--days`` days) and
``dashboard.htpasswd`` (nginx basic authentication, ``$apr1$`` hash). The password comes from $DASHBOARD_PASSWORD;
without it a random one is generated and written to ``dashboard-password.txt`` next to the other files, never
printed. For production, put a certificate from a real CA in the directory instead of the self-signed one.

The certificate is made with a local ``openssl`` when there is one, otherwise inside the pinned python:3.12-slim
image (Docker).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON_IMAGE = "python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b"
ITOA64 = "./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def apr1(password: str, salt: str) -> str:
    """Apache's MD5-based ``$apr1$`` crypt, which nginx checks natively (no system crypt needed)."""
    pw, salt_b = password.encode(), salt.encode()[:8]
    ctx = pw + b"$apr1$" + salt_b
    final = hashlib.md5(pw + salt_b + pw).digest()
    for remaining in range(len(pw), 0, -16):
        ctx += final[:min(16, remaining)]
    i = len(pw)
    while i:
        ctx += b"\x00" if i & 1 else pw[:1]
        i >>= 1
    final = hashlib.md5(ctx).digest()
    for i in range(1000):
        step = pw if i & 1 else final
        if i % 3:
            step += salt_b
        if i % 7:
            step += pw
        step += final if i & 1 else pw
        final = hashlib.md5(step).digest()

    def to64(value: int, count: int) -> str:
        out = ""
        for _ in range(count):
            out += ITOA64[value & 0x3F]
            value >>= 6
        return out

    encoded = "".join(to64((final[a] << 16) | (final[b] << 8) | final[c], 4)
                      for a, b, c in ((0, 6, 12), (1, 7, 13), (2, 8, 14), (3, 9, 15), (4, 10, 5)))
    return f"$apr1${salt_b.decode()}${encoded}{to64(final[11], 2)}"


def make_certificate(directory: Path, days: int) -> None:
    command = ["req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", str(days), "-subj", "/CN=localhost",
               "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1"]
    if shutil.which("openssl"):
        subprocess.run(["openssl", *command, "-keyout", str(directory / "server.key"),
                        "-out", str(directory / "server.crt")], check=True, capture_output=True)
        return
    subprocess.run(["docker", "run", "--rm", "-v", f"{directory.resolve().as_posix()}:/out", PYTHON_IMAGE,
                    "openssl", *command, "-keyout", "/out/server.key", "-out", "/out/server.crt"],
                   check=True, capture_output=True, env={**os.environ, "MSYS_NO_PATHCONV": "1"})


def make_secrets(directory: Path, user: str = "admin", password: str | None = None, days: int = 30) -> Path:
    """Create the files; returns the htpasswd path. A missing password is generated and saved, not printed."""
    directory.mkdir(parents=True, exist_ok=True)
    make_certificate(directory, days)
    if password is None:
        password = secrets.token_urlsafe(18)
        secret_file = directory / "dashboard-password.txt"
        secret_file.write_text(password + "\n", encoding="utf-8")
        secret_file.chmod(0o600)
    htpasswd = directory / "dashboard.htpasswd"
    salt = "".join(secrets.choice(ITOA64) for _ in range(8))
    htpasswd.write_text(f"{user}:{apr1(password, salt)}\n", encoding="utf-8")
    return htpasswd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default=str(ROOT / "config" / "tls"))
    parser.add_argument("--user", default="admin", help="dashboard user name")
    parser.add_argument("--days", type=int, default=30, help="certificate validity")
    args = parser.parse_args()
    directory = Path(args.dir)
    password = os.environ.get("DASHBOARD_PASSWORD") or None
    make_secrets(directory, args.user, password, args.days)
    where = "from $DASHBOARD_PASSWORD" if password else f"generated, saved in {directory / 'dashboard-password.txt'}"
    print(f"wrote server.crt, server.key and dashboard.htpasswd to {directory} (user {args.user!r}, password {where})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
