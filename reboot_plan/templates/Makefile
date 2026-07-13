# RedRHex core-reboot Makefile requirements — P0 design template
#
# This is intentionally not an executable Makefile yet. P0 implements foundation/runtime
# checks and freezes the remaining target interfaces; G0 implements and dry-runs the
# diagnostic orchestrator. Do not copy the old draft
# from commit a795c06: it hard-coded a home path, changed existing panel defaults, and
# put baseline capture before simulation validation.

# Runtime roots must be caller-configurable, never embedded as /home/<user> paths:
#
#   ISAACLAB_ROOT ?= <required for Isaac targets>
#   ISAACSIM_ROOT ?= <required for provenance checks>
#   DEVICE        ?= cuda:0
#   TASK          ?= Template-Redrhex-Direct-v0
#
# Required target order and behavior:
#
#   verify-foundation
#       Safe complete CPU collection, CPU tests, ignored-test probes, frozen-tree guard.
#
#   verify-runtime
#       Check source/runtime SHAs, dirty-state policy, Isaac Sim build, Python/packages.
#
#   validate-sim-gravity
#       Interface/refusal/artifact schema fixed in P0; executable orchestrator begins at
#       G0, then runs G0–G10 by their prerequisite graph and records C1 separately.
#
#   capture-golden BASELINE_ID=<id>
#       Refuse to run unless the linked P1 result is PASS. Produce a small tracked
#       fixture plus full local artifacts and hashes.
#
#   smoke-reboot NUM_ENVS=<explicit> MAX_ITERATIONS=<explicit> SEED=<explicit>
#       Never silently replace local panel defaults (4 envs/1 iteration) or Remote Web
#       defaults (4 envs/8 iterations).
#
#   train-ref BASELINE_ID=<id> SEEDS=<explicit>
#       Refuse to run before P1/P2 manifest checks; dry-run prints the complete commands.
#
#   test-contract | test-core | test-golden | test-adapter | preflight
#       Implement the tiers in reboot_plan/05_testing_and_ci.md. CPU targets must not
#       initialize Isaac. Frozen consumer tests remain in their existing directories.
#
#   help
#       Document required environment variables and which targets use the GPU.
#
# Every mutating/long GPU target needs a non-GPU dry-run mode and writes the exact command
# into its manifest. ROS generation is intentionally absent because ros2_ws is frozen.
