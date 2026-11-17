"""Routing: which model tier answers a request."""

from __future__ import annotations

import logging
import math

from gateway.config import GatewayConfig, resolve_path
from gateway.routing.base import Router, RouteScore, choose_tier, routing_text
from gateway.routing.rules import RulesRouter

__all__ = ["RouteScore", "Router", "RulesRouter", "build_router", "choose_tier", "routing_text"]

log = logging.getLogger("gateway.routing")


def build_router(config: GatewayConfig) -> Router:
    """The configured router; the rule-based one when the classifier can't be loaded."""
    if config.routing.router == "rules":
        return RulesRouter()
    path = resolve_path(config.routing.classifier_path)
    if not path.exists():
        log.warning("router file %s not found (train it with scripts/train_router.py); using rules", path)
        return RulesRouter(note="no trained classifier yet")
    try:
        from gateway.routing.classifier import ClassifierRouter

        router = ClassifierRouter.load(path)
        if not math.isclose(router.selected_threshold, config.routing.classifier_threshold, abs_tol=1e-12):
            raise ValueError(
                f"router selected threshold {router.selected_threshold} does not match "
                f"routing.classifier_threshold {config.routing.classifier_threshold}"
            )
        return router
    except Exception as e:  # a broken file or a missing embedder must not stop the gateway
        log.warning("could not load router %s (%s: %s); using rules", path, type(e).__name__, e)
        return RulesRouter(note=f"classifier failed to load: {type(e).__name__}")
