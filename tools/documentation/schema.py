"""Documentation metadata constants."""

REQUIRED_FIELDS = (
    "id",
    "title",
    "lang",
    "audience",
    "type",
    "status",
    "owner",
    "last_reviewed",
)

ALLOWED_VALUES = {
    "lang": {"en", "zh-TW"},
    "audience": {"operator", "developer", "shared"},
    "owner": {"project", "core", "training", "panel", "deployment", "sim2real", "reward-agent"},
    "type": {
        "index", "tutorial", "how-to", "reference", "explanation", "safety",
        "troubleshooting", "decision", "design", "plan", "roadmap", "release",
        "experiment-summary", "audit",
    },
}

STATUS_BY_TYPE = {
    "index": {"draft", "active", "deprecated"},
    "tutorial": {"draft", "active", "deprecated"},
    "how-to": {"draft", "active", "deprecated"},
    "reference": {"draft", "active", "deprecated"},
    "explanation": {"draft", "active", "deprecated"},
    "safety": {"draft", "active", "deprecated"},
    "troubleshooting": {"draft", "active", "deprecated"},
    "decision": {"accepted", "superseded"},
    "design": {"proposed", "approved", "implemented", "rejected", "superseded"},
    "plan": {"draft", "active", "blocked", "completed", "cancelled"},
    "roadmap": {"active"},
    "release": {"published"},
    "experiment-summary": {"published"},
    "audit": {"published"},
}
