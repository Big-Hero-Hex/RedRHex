"""Validate pull-request documentation-impact declarations."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Sequence


ALLOWED_IMPACTS = (
    "none",
    "operator",
    "developer",
    "shared",
    "release",
    "experiment",
)
HTML_COMMENT = re.compile(r"<!--.*?(?:-->|\Z)", re.DOTALL)


def validate_declaration(body: str) -> list[str]:
    """Return deterministic human-readable declaration errors."""

    errors: list[str] = []
    body = HTML_COMMENT.sub("", body)
    impact_values = [
        line.removeprefix("Docs impact:").strip()
        for line in body.splitlines()
        if line.startswith("Docs impact:")
    ]
    reason_values = [
        line.removeprefix("Docs reason:").strip()
        for line in body.splitlines()
        if line.startswith("Docs reason:")
    ]
    if not impact_values:
        errors.append("missing Docs impact field")
    elif len(impact_values) > 1:
        errors.append("duplicate Docs impact field")
    elif impact_values[0] not in ALLOWED_IMPACTS:
        errors.append(
            "Docs impact must be one of: none, operator, developer, shared, "
            "release, experiment"
        )
    if not reason_values:
        errors.append("missing Docs reason field")
    elif len(reason_values) > 1:
        errors.append("duplicate Docs reason field")
    elif HTML_COMMENT.sub("", reason_values[0]).strip() in (
        "",
        "<required explanation>",
    ):
        errors.append("Docs reason must contain non-whitespace prose")
    return errors


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.documentation.pr_declaration",
        allow_abbrev=False,
    )
    parser.add_argument("--event-json", action="append", required=True, metavar="PATH")
    return parser


def _report_failure(errors: Sequence[str]) -> int:
    for error in errors:
        print(error, file=sys.stderr)
    print(
        f"documentation impact declaration failed ({len(errors)} errors)",
        file=sys.stderr,
    )
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if len(arguments.event_json) != 1:
        parser.error("argument --event-json: may not be repeated")
    try:
        event_text = Path(arguments.event_json[0]).read_text(encoding="utf-8")
    except UnicodeError:
        return _report_failure(["GitHub event JSON is not valid UTF-8"])
    except OSError:
        return _report_failure(["unable to read GitHub event JSON"])
    try:
        event = json.loads(event_text)
    except ValueError:
        return _report_failure(["GitHub event JSON is malformed"])
    pull_request = event.get("pull_request") if isinstance(event, dict) else None
    if not isinstance(pull_request, dict):
        return _report_failure(
            ["GitHub event is missing a pull_request object"]
        )
    body = pull_request.get("body")
    if not isinstance(body, str):
        return _report_failure(["pull_request.body must be a string"])
    errors = validate_declaration(body)
    if not errors:
        print("documentation impact declaration passed")
        return 0
    return _report_failure(errors)


if __name__ == "__main__":
    raise SystemExit(main())
