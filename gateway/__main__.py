"""Run the gateway: ``python -m gateway`` (GATEWAY_HOST / GATEWAY_PORT, default 127.0.0.1:8000)."""

import logging
import os

import uvicorn


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run("gateway.app:create_app", factory=True, host=os.environ.get("GATEWAY_HOST", "127.0.0.1"),
                port=int(os.environ.get("GATEWAY_PORT", "8000")))


if __name__ == "__main__":
    main()
