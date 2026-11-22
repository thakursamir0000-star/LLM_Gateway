"""Run the gateway: ``python -m gateway`` (GATEWAY_HOST / GATEWAY_PORT, default 127.0.0.1:8000)."""

import os

import uvicorn

from gateway.logs import configure_logging


def main() -> None:
    configure_logging()
    uvicorn.run("gateway.app:create_app", factory=True, host=os.environ.get("GATEWAY_HOST", "127.0.0.1"),
                port=int(os.environ.get("GATEWAY_PORT", "8000")), log_config=None)


if __name__ == "__main__":
    main()
