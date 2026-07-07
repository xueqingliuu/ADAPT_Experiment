"""Framework-facing weekly query policy.

This handler implements the simple query rule used by the current experiment
baselines:

    I_2 = 1
    I_w ~ Bernoulli(0.5), for all other weeks

Policy knobs (``query_prob``, ``force_query_week``, ``seed``, ``burn_in_days``)
live in ``config.json``. ``model_params`` only stores the RNG stream state needed
to continue the Bernoulli sequence across requests.

``make_state`` derives the 1-indexed study week ``w`` from the request context
(decision datetime + timezone) and participant enrollment metadata from
``db_reader.get_retrieved_data()``. Day-of-week and decision-slot fields belong
in the RL handler, not here.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from RL_agents.base import (
    ActionHandlerBase,
    ActionRecord,
    ActionResult,
    DBReader,
    StateResult,
    UpdatePrepResult,
    UpdateResult,
)


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _stable_seed(user_id: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(str(user_id).encode("utf-8"), digest_size=4).digest(),
        "little",
    )


def _week_from_state(state: dict) -> int:
    """Return a 1-indexed query week from a pre-built state."""
    if "w" in state and state["w"] is not None:
        return int(state["w"])
    if "k" in state and state["k"] is not None:
        return int(state["k"]) + 1
    if "rl_week" in state and state["rl_week"] is not None:
        return int(state["rl_week"]) + 1
    if "week" in state and state["week"] is not None:
        return int(state["week"])
    raise KeyError("Query state must include w or calendar-derived week indices")


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        return datetime.fromisoformat(text)
    raise TypeError(f"Unsupported datetime value: {value!r}")


def _context_datetime(context: dict) -> datetime | None:
    for key in ("decision_time", "decisionTime", "timestamp", "request_time"):
        if key in context and context[key] is not None:
            return _parse_datetime(context[key])
    return None


def _context_timezone(context: dict) -> str:
    for key in ("timezone", "time_zone", "tz"):
        value = context.get(key)
        if value:
            return str(value)
    raise KeyError("Query context must include timezone, time_zone, or tz")


def _participant_info(db_reader: DBReader) -> dict:
    data = db_reader.get_retrieved_data() or {}
    info = (
        data.get("participantinfo")
        or data.get("participant_info")
        or data.get("ParticipantInfo")
    )
    if info is None:
        raise KeyError(
            "db_reader.get_retrieved_data() must include participantinfo"
        )
    if isinstance(info, list):
        if not info:
            raise KeyError("participantinfo list is empty")
        info = info[0]
    return dict(info)


def _study_start_date(participant: dict) -> date:
    for key in (
        "study_start_date",
        "enroll_date",
        "start_date",
        "StudyStartDate",
        "date_min",
    ):
        value = participant.get(key)
        if value is None:
            continue
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        return _parse_datetime(value).date()
    raise KeyError(
        "participantinfo must include study_start_date, enroll_date, or date_min"
    )


def _study_week(
    study_start: date,
    local_decision_date: date,
    *,
    burn_in_days: int,
) -> int:
    """Return the 0-indexed study week for ``local_decision_date``."""
    anchor = study_start if burn_in_days <= 0 else study_start.fromordinal(
        study_start.toordinal() + burn_in_days
    )
    day = (local_decision_date - anchor).days
    if day < 0:
        raise ValueError(
            f"Decision date {local_decision_date} is before study anchor {anchor}"
        )

    start_dow = anchor.weekday()
    return (day + start_dow) // 7 + int(start_dow == 0)


class Query(ActionHandlerBase):
    """Weekly query decision handler with the ``ActionHandlerBase`` API."""

    action_type = "weekly_query"

    def __init__(self, config: dict | None = None):
        self.config = load_config() if config is None else config

    def _rng(self, model_params: dict) -> np.random.Generator:
        rng = np.random.default_rng()
        rng_state = model_params.get("rng_state")
        if rng_state is not None:
            rng.bit_generator.state = rng_state
        return rng

    def get_initial_model_params(self, user_id: str) -> dict:
        seed = self.config.get("seed")
        rng = np.random.default_rng(
            _stable_seed(user_id) if seed is None else int(seed)
        )
        return {
            "version": 1,
            "user_id": str(user_id),
            "rng_state": rng.bit_generator.state,
        }

    def make_state(
        self,
        user_id: str,
        context: dict,
        db_reader: DBReader,
    ) -> StateResult:
        state = dict(context or {})
        decision_dt = _context_datetime(state)

        if decision_dt is not None:
            tz = ZoneInfo(_context_timezone(state))
            if decision_dt.tzinfo is None:
                decision_dt = decision_dt.replace(tzinfo=tz)
            local_dt = decision_dt.astimezone(tz)

            participant = _participant_info(db_reader)
            study_start = _study_start_date(participant)
            burn_in_days = int(self.config.get("burn_in_days", 6))
            week = _study_week(
                study_start,
                local_dt.date(),
                burn_in_days=burn_in_days,
            )
            state.update(
                {
                    "w": week + 1,
                    "decision_time_local": local_dt.isoformat(),
                }
            )
        else:
            state["w"] = _week_from_state(state)

        return StateResult(state=state)

    def get_action(
        self,
        user_id: str,
        state: dict,
        model_params: dict,
    ) -> ActionResult:
        w = _week_from_state(state or {})
        force_week = int(self.config.get("force_query_week", 2))
        query_prob = float(self.config.get("query_prob", 0.5))

        rng = self._rng(model_params)
        if w == force_week:
            action = 1
            action_prob = 1.0
        else:
            action_prob = query_prob
            action = int(rng.binomial(1, action_prob))

        model_params["rng_state"] = rng.bit_generator.state
        return ActionResult(
            action=action,
            action_prob=float(action_prob),
            random_state={"rng_state": rng.bit_generator.state},
        )

    def prepare_for_update(
        self,
        user_id: str,
        db_reader: DBReader,
    ) -> UpdatePrepResult:
        return UpdatePrepResult(rewards={})

    def update(
        self,
        user_id: str,
        model_params: dict,
        actions: list[ActionRecord],
        rewards: dict[int, float],
        db_reader: DBReader,
    ) -> UpdateResult:
        return UpdateResult(model_params=model_params)


QueryAgent = Query
