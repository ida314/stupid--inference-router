"""Structured logging for scheduler decisions and lifecycle events.

The README asks for one specific property: read a log line, understand why the scheduler
did what it did. So a decision is never logged as "switching to translate" — it's logged
with every candidate's score and the reason that outranked the others.
"""

from __future__ import annotations

import json
import logging
import sys

from sir.types import Decision, DecisionKind, EngineEvent, EventKind

logger = logging.getLogger("sir")

_LOUD_EVENTS = {
    EventKind.LOAD_START,
    EventKind.LOAD_END,
    EventKind.LOAD_FAILED,
    EventKind.UNLOAD,
    EventKind.DRAIN_START,
    EventKind.DRAIN_END,
    EventKind.CRASH,
    EventKind.FAILED,
}


def configure(level: str = "info") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    logger.handlers = [handler]
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False


def log_decision(decision: Decision) -> None:
    payload = {
        "event": "decision",
        "kind": decision.kind.value,
        "reason": decision.reason.value,
        "resident": decision.resident,
        "target": decision.target,
        "at": round(decision.at, 3),
        "scores": {
            score.model: {
                "score": None if score.score == float("-inf") else round(score.score, 3),
                "depth": score.depth,
                "oldest_wait": round(score.oldest_wait, 3),
                "switch_cost": round(score.switch_cost, 3),
                "eligible": score.eligible,
                # Says why a model scored zero or was skipped, so a reader never has to
                # guess whether a low score meant "lost" or "wasn't in the running".
                **({"note": score.note} if score.note else {}),
            }
            for score in decision.scores
        },
    }
    level = logging.INFO if decision.kind is DecisionKind.SWITCH else logging.DEBUG
    logger.log(level, json.dumps(payload))


def log_event(event: EngineEvent) -> None:
    payload = {
        "event": event.kind.value,
        "model": event.model,
        "at": round(event.at, 3),
    }
    if event.duration is not None:
        payload["duration"] = round(event.duration, 3)
    if event.detail:
        payload["detail"] = event.detail
    level = logging.INFO if event.kind in _LOUD_EVENTS else logging.DEBUG
    logger.log(level, json.dumps(payload))
