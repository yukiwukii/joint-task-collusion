"""Two-agent tool protocol internals."""

MEMORY_SCOPE = "cross-episode"

# Default communication rounds; all configured rounds run.
DEFAULT_MAX_ROUNDS = 5


# Separate task and delivery budgets so task complexity does not reduce channel retries.
TASK_ATTEMPTS = 15
COMMUNICATION_ATTEMPTS = 3

# Per-agent retries for malformed verdict submissions.
VERDICT_ATTEMPTS = 3

# Safety cap on free workspace_log-only turns per agent per budgeted slot (one task
# phase or one communication round; verdict turns are never free). Past it, such turns cost
# a normal attempt, so a looping agent still ends.
WORKSPACE_FREE_TURNS = 50
