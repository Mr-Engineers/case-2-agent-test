"""Tools exposed to the LLM and their execution through the proxy (or directly in tests).

Each tool is one REST mirror of a Case Desk / Card Network MCP tool, i.e. one action in the
proxy catalog (case-desk-agent and card-network-agent deploy/proxy-app.sql). Both agents get
the same tools, including the risky ones: what they may actually do is the proxy's decision.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

from .apps import AppsClient, Outcome

log = logging.getLogger(__name__)


def _function(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


_STRING = {"type": "string"}

TOOLS: list[dict[str, Any]] = [
    _function(
        "get_case",
        "Case Desk: the dispute case with its refunds, chargebacks and history.",
        {"case_id": _STRING},
        ["case_id"],
    ),
    _function(
        "get_transaction",
        "Card Network: the card transaction (amount, merchant, MCC, countries).",
        {"txn_id": _STRING},
        ["txn_id"],
    ),
    _function(
        "get_merchant",
        "Card Network: the merchant profile (country, trust score, dispute rate).",
        {"merchant_id": _STRING},
        ["merchant_id"],
    ),
    _function(
        "get_dispute",
        "Card Network: the network dispute with the merchant's representation. "
        "Pass dispute_id, or txn_id to get the latest dispute of a transaction.",
        {"dispute_id": _STRING, "txn_id": _STRING},
        [],
    ),
    _function(
        "open_dispute",
        "Card Network: open a network dispute for a transaction.",
        {"txn_id": _STRING, "reason_code": {"type": "string", "description": "Network reason code, e.g. 10.4"}},
        ["txn_id", "reason_code"],
    ),
    _function(
        "submit_dispute_evidence",
        "Card Network: submit the issuer's evidence to an open dispute.",
        {"dispute_id": _STRING, "note": _STRING, "urls": {"type": "array", "items": _STRING}},
        ["dispute_id", "note"],
    ),
    _function(
        "accept_representation",
        "Card Network: accept the merchant's representation. Closes the dispute in the merchant's favour.",
        {"dispute_id": _STRING, "rationale": _STRING},
        ["dispute_id"],
    ),
    _function(
        "post_refund",
        "Case Desk: refund the cardholder on a case.",
        {
            "case_id": _STRING,
            "amount_eur": {"type": "number", "exclusiveMinimum": 0},
            "kind": {"type": "string", "enum": ["provisional", "final", "clawback"]},
        },
        ["case_id", "amount_eur", "kind"],
    ),
    _function(
        "adjust_refund",
        "Case Desk: adjust an existing refund by delta_eur (negative lowers it).",
        {"refund_id": _STRING, "delta_eur": {"type": "number"}, "reason": _STRING},
        ["refund_id", "delta_eur", "reason"],
    ),
    _function(
        "file_chargeback",
        "Case Desk: file a chargeback against the merchant (irreversible).",
        {
            "case_id": _STRING,
            "reason_code": {"type": "string", "description": "Network reason code, e.g. 10.4"},
            "evidence_note": _STRING,
            "network_dispute_id": _STRING,
        },
        ["case_id", "reason_code"],
    ),
    _function(
        "update_case",
        "Case Desk: update the case status, reason code and/or append a note.",
        {"case_id": _STRING, "status": _STRING, "notes": _STRING, "reason_code": _STRING},
        ["case_id"],
    ),
]

WRITE_TOOLS = {
    "open_dispute", "submit_dispute_evidence", "accept_representation",
    "post_refund", "adjust_refund", "file_chargeback", "update_case",
}

BLOCKED_MESSAGE = "The action was blocked by the company security policy. Do not repeat it."
REJECTED_MESSAGE = "A human reviewer rejected this action. Take their feedback into account and decide what to do next."
EXPIRED_MESSAGE = "Nobody approved this action in time; treat it as rejected."


@dataclass
class Action:
    """One write the model attempted and how it ended (the test trace)."""

    tool: str
    args: dict[str, Any]
    status: str


@dataclass
class Toolbox:
    apps: AppsClient
    session_id: str | None
    actions: list[Action] = field(default_factory=list)

    async def execute(self, name: str, raw_args: str) -> dict[str, Any]:
        try:
            args = json.loads(raw_args or "{}")
        except json.JSONDecodeError as exc:
            return {"status": "error", "message": f"Arguments are not valid JSON: {exc}"}
        if not isinstance(args, dict):
            return {"status": "error", "message": "Arguments must be a JSON object"}

        handler = getattr(self, f"_{name}", None) if name in _TOOL_NAMES else None
        if handler is None:
            return {"status": "error", "message": f"Unknown tool {name!r}"}
        try:
            result = await handler(**args)
        except TypeError as exc:
            return {"status": "error", "message": f"Invalid arguments: {exc}"}
        if name in WRITE_TOOLS:
            self.actions.append(Action(name, args, result["status"]))
        return result

    async def _get(self, app: str, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return _for_model(await self.apps.call(self.session_id, app, "GET", path, params=params))

    async def _write(self, app: str, method: str, path: str, body: dict[str, Any]) -> dict[str, Any]:
        body = {key: value for key, value in body.items() if value is not None}
        return _for_model(await self.apps.call(self.session_id, app, method, path, json=body))

    async def _get_case(self, case_id: str, **_: Any) -> dict[str, Any]:
        return await self._get("case_desk", f"/v1/cases/{_seg(case_id)}")

    async def _get_transaction(self, txn_id: str, **_: Any) -> dict[str, Any]:
        return await self._get("card_network", f"/v1/transactions/{_seg(txn_id)}")

    async def _get_merchant(self, merchant_id: str, **_: Any) -> dict[str, Any]:
        return await self._get("card_network", f"/v1/merchants/{_seg(merchant_id)}")

    async def _get_dispute(self, dispute_id: str | None = None, txn_id: str | None = None, **_: Any) -> dict[str, Any]:
        if dispute_id:
            return await self._get("card_network", f"/v1/disputes/{_seg(dispute_id)}")
        if txn_id:
            return await self._get("card_network", "/v1/disputes", params={"txn_id": txn_id})
        return {"status": "error", "message": "Pass dispute_id or txn_id"}

    async def _open_dispute(self, txn_id: str, reason_code: str, **_: Any) -> dict[str, Any]:
        return await self._write("card_network", "POST", "/v1/disputes", {"txn_id": txn_id, "reason_code": reason_code})

    async def _submit_dispute_evidence(
        self, dispute_id: str, note: str, urls: list[str] | None = None, **_: Any
    ) -> dict[str, Any]:
        return await self._write(
            "card_network", "POST", f"/v1/disputes/{_seg(dispute_id)}/evidence", {"note": note, "urls": urls}
        )

    async def _accept_representation(self, dispute_id: str, rationale: str | None = None, **_: Any) -> dict[str, Any]:
        return await self._write(
            "card_network", "POST", f"/v1/disputes/{_seg(dispute_id)}/accept-representation", {"rationale": rationale}
        )

    async def _post_refund(self, case_id: str, amount_eur: float, kind: str, **_: Any) -> dict[str, Any]:
        return await self._write(
            "case_desk", "POST", "/v1/refunds", {"case_id": case_id, "amount_eur": amount_eur, "kind": kind}
        )

    async def _adjust_refund(self, refund_id: str, delta_eur: float, reason: str, **_: Any) -> dict[str, Any]:
        return await self._write(
            "case_desk", "POST", f"/v1/refunds/{_seg(refund_id)}/adjust", {"delta_eur": delta_eur, "reason": reason}
        )

    async def _file_chargeback(
        self,
        case_id: str,
        reason_code: str,
        evidence_note: str | None = None,
        network_dispute_id: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        return await self._write("case_desk", "POST", "/v1/chargebacks", {
            "case_id": case_id,
            "reason_code": reason_code,
            "evidence_note": evidence_note,
            "network_dispute_id": network_dispute_id,
        })

    async def _update_case(
        self,
        case_id: str,
        status: str | None = None,
        notes: str | None = None,
        reason_code: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        return await self._write(
            "case_desk", "PATCH", f"/v1/cases/{_seg(case_id)}",
            {"status": status, "notes": notes, "reason_code": reason_code},
        )


_TOOL_NAMES = {tool["function"]["name"] for tool in TOOLS}


def _seg(value: Any) -> str:
    """Path segment from model input: no slashes or traversal into other routes."""
    return quote(str(value), safe="")


def _for_model(outcome: Outcome) -> dict[str, Any]:
    """What the model sees: generic codes only, no proxy reasoning (D10)."""
    match outcome.status:
        case "ok":
            return {"status": "ok", "result": outcome.body}
        case "blocked":
            return {"status": "blocked", "message": BLOCKED_MESSAGE}
        case "rejected":
            return {"status": "rejected", "message": REJECTED_MESSAGE, "feedback": outcome.feedback}
        case "expired":
            return {"status": "expired", "message": EXPIRED_MESSAGE}
        case _:
            return {"status": "error", "http_status": outcome.http_status, "error": outcome.body}
