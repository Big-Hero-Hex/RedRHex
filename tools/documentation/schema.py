REQUIRED_FIELDS = frozenset({"id", "title", "lang", "audience", "type", "status", "owner", "last_reviewed"})

LANGUAGES = frozenset({"en", "zh-TW"})
AUDIENCES = frozenset({"operator", "developer", "shared"})
OWNERS = frozenset({"project", "core", "training", "panel", "deployment", "sim2real", "reward-agent"})
KNOWLEDGE_TYPES = frozenset({"index", "tutorial", "how-to", "reference", "explanation", "safety", "troubleshooting"})
STATUS_BY_TYPE = {
    **{kind: frozenset({"draft", "active", "deprecated"}) for kind in KNOWLEDGE_TYPES},
    "decision": frozenset({"accepted", "superseded"}),
    "design": frozenset({"proposed", "approved", "implemented", "rejected", "superseded"}),
    "plan": frozenset({"draft", "active", "blocked", "completed", "cancelled"}),
    "roadmap": frozenset({"active"}), "release": frozenset({"published"}),
    "experiment-summary": frozenset({"published"}), "audit": frozenset({"published"}),
}

CENTRAL_SECTIONS = {
    "operators": ("operator", frozenset({"tutorial", "how-to", "reference", "explanation", "safety", "troubleshooting"})),
    "developers": ("developer", frozenset({"tutorial", "how-to", "reference", "explanation", "safety", "troubleshooting"})),
    "reference": ("shared", frozenset({"reference"})), "decisions": ("developer", frozenset({"decision"})),
    "designs": ("developer", frozenset({"design"})), "plans": ("developer", frozenset({"plan"})),
    "roadmap": ("shared", frozenset({"roadmap"})), "releases": ("shared", frozenset({"release"})),
    "research": ("developer", frozenset({"experiment-summary", "audit", "explanation"})),
    "governance": ("developer", frozenset({"reference"})),
}
