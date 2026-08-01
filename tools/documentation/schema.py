"""Governance schema constants used by the documentation validator."""

REQUIRED_FIELDS = frozenset(
    {
        "id",
        "title",
        "lang",
        "audience",
        "type",
        "status",
        "owner",
        "last_reviewed",
    }
)

ENUM_FIELDS = {
    "lang": frozenset({"en", "zh-TW"}),
    "audience": frozenset({"operator", "developer", "shared"}),
    "owner": frozenset(
        {"project", "core", "training", "panel", "deployment", "sim2real", "reward-agent"}
    ),
    "type": frozenset(
        {
            "index",
            "tutorial",
            "how-to",
            "reference",
            "explanation",
            "safety",
            "troubleshooting",
            "decision",
            "design",
            "plan",
            "roadmap",
            "release",
            "experiment-summary",
            "audit",
        }
    ),
}

_KNOWLEDGE_STATUSES = frozenset({"draft", "active", "deprecated"})
STATUS_BY_TYPE = {
    "index": _KNOWLEDGE_STATUSES,
    "tutorial": _KNOWLEDGE_STATUSES,
    "how-to": _KNOWLEDGE_STATUSES,
    "reference": _KNOWLEDGE_STATUSES,
    "explanation": _KNOWLEDGE_STATUSES,
    "safety": _KNOWLEDGE_STATUSES,
    "troubleshooting": _KNOWLEDGE_STATUSES,
    "decision": frozenset({"accepted", "superseded"}),
    "design": frozenset({"proposed", "approved", "implemented", "rejected", "superseded"}),
    "plan": frozenset({"draft", "active", "blocked", "completed", "cancelled"}),
    "roadmap": frozenset({"active"}),
    "release": frozenset({"published"}),
    "experiment-summary": frozenset({"published"}),
    "audit": frozenset({"published"}),
}

KNOWLEDGE_TYPES = frozenset(
    {"index", "tutorial", "how-to", "reference", "explanation", "safety", "troubleshooting"}
)
CENTRAL_LOCATION_RULES = {
    "operators": ("operator", KNOWLEDGE_TYPES),
    "developers": ("developer", KNOWLEDGE_TYPES),
    "reference": ("shared", frozenset({"index", "reference"})),
    "decisions": ("developer", frozenset({"index", "decision"})),
    "designs": ("developer", frozenset({"index", "design"})),
    "plans": ("developer", frozenset({"index", "plan"})),
    "roadmap": ("shared", frozenset({"index", "roadmap"})),
    "releases": ("shared", frozenset({"index", "release"})),
    "research": (
        "developer",
        frozenset({"index", "experiment-summary", "audit", "explanation"}),
    ),
    "governance": ("developer", frozenset({"index", "reference"})),
}
