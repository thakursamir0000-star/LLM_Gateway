# Security policy

## Reporting a vulnerability

Please report security problems privately to the repository owner (GitHub security advisories), not in a public
issue. Include the affected version or commit and how to reproduce it.

## Keys and secrets

- **Client keys**: `GATEWAY_API_KEYS` and/or the file named by `GATEWAY_API_KEYS_FILE` (a mounted secret), one
  `app:key` entry per line or comma. Store keys hashed: `python scripts/make_api_key.py <app> --append <file>`
  writes `app:sha256:<hex>` and prints the key once for the client; the gateway compares SHA-256 digests in
  constant time, so a leaked keys file holds no usable key.
- **Rotation without downtime**: add the new entry next to the old one, restart the replicas one at a time, move the
  client to the new key, delete the old entry and restart again. A revoked key gets 401 on the next restart.
- **Admin key**: `GATEWAY_ADMIN_KEY` may also be given as `sha256:<hex>` (lower case). The admin API is never
  published by the HTTPS proxy.
- **Provider keys** (`GROQ_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY`), the database password, the Grafana
  password and the metrics token belong in the deployment's secret store, never in the repository; `.env` is
  git-ignored. Rotate a provider key at the provider, update the secret, then restart the replicas one at a time.

## Dependencies (supply chain)

- **Inputs and locks.** `requirements.txt` (the service), `requirements-dev.txt` (tests and tools) and
  `requirements-lab.txt` (router lab) hold the allowed version ranges. `requirements.lock` and `requirements-dev.lock`
  pin every package, including transitive ones, to one version with SHA-256 hashes, for Python 3.11+ on all platforms.
  The Docker image installs `requirements.lock` and CI installs `requirements-dev.lock`, both with `--require-hashes`.
- **Refreshing the locks** (after editing a requirements file, or to take updates), with the same tool versions CI uses:

  ```bash
  docker run --rm -v "$PWD:/src" -w /src python:3.12-slim sh -c 'pip install -q "uv==0.13.9" && \
    uv pip compile requirements.txt --universal --python-version 3.11 --generate-hashes --quiet -o requirements.lock && \
    uv pip compile requirements-dev.txt --universal --python-version 3.11 --generate-hashes --quiet -o requirements-dev.lock'
  ```

  uv keeps the existing pins; add `--upgrade` (or `--upgrade-package <name>`) to move them, then run the full test
  suite and review the changes before committing. Don't upgrade everything blindly: read the changelogs of major
  versions.
- **Base image.** The Dockerfile pins `python:3.12-slim` by digest. Dependabot (`.github/dependabot.yml`) proposes new
  digests and GitHub Actions updates weekly. It doesn't regenerate the uv locks; the audit below covers them.
- **Vulnerability audit.** `.github/workflows/dependency-audit.yml` runs on every push and pull request and every
  Monday. It fails when a lock doesn't match its requirements file, or when `pip-audit` finds a known vulnerability
  (PyPI advisory database) in either lock.

### Severity policy

Any known vulnerability in a locked package fails the audit, whatever its severity: pip-audit's database has no
reliable severity for every advisory, and the dependency set is small enough to fix or justify each finding. Fix by
upgrading the package in the lock. When a fix isn't available or the vulnerable code isn't reachable, add the advisory
to `.github/pip-audit-exceptions.txt` with the package, the date and the reason; exceptions are reviewed at every lock
refresh. None are accepted today.

### Not locked

- The optional `WITH_EMBEDDINGS=true` image build installs `sentence-transformers` and CPU torch from version ranges.
- `requirements-lab.txt` (offline router-lab scripts, not part of the service or the image).

Last audit: 2026-11-22, pip-audit 2.10.1, no known vulnerabilities in either lock.
