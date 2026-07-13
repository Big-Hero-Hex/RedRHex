# RedRHex Makefile — canonical verification entry points (copy to repo root, Phase 1.1)
# Every target must work from a fresh shell with only `conda activate env_isaaclab_bin`.

ISAACLAB   := /home/lab_user1/isaac_lab_ws/IsaacLab/isaaclab.sh
TASK       := Template-Redrhex-Direct-v0
SMOKE_ENVS := 128
SMOKE_ITER := 5
SEED       ?= 42
EXP        ?=

.PHONY: lint test-fast test-contract test-golden test smoke preflight contract \
        train train-ref gpu-guard help

help: ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

lint: ## ruff format check + lint (seconds, CPU)
	ruff format --check . && ruff check .

test-fast: ## tier 1: pure-CPU unit tests, no isaaclab (<30s)
	pytest tests/unit -m fast -q

test-contract: ## tier 2: contract/layout/generated-file parity (seconds)
	pytest tests/contract -q

test-golden: ## tier 3: replay recorded baseline tensors through core modules (<1min)
	pytest tests/golden -m golden -q

test: lint test-fast test-contract test-golden ## all CPU tiers (= what CI runs)

gpu-guard: ## refuse GPU targets if a training run is live
	@python scripts/diagnostics/gpu_guard.py  # exits 1 if panel/nvidia-smi shows a live run

smoke: gpu-guard ## tier 4: short headless train, env integration (~5min, GPU)
	$(ISAACLAB) -p scripts/rsl_rl/train.py --task $(TASK) --headless \
	  --num_envs $(SMOKE_ENVS) --max_iterations $(SMOKE_ITER) --seed $(SEED)

preflight: test smoke ## full local gate before merging to main
	pytest tests/sim -m isaac -q
	python scripts/diagnostics/onnx_equivalence.py --golden baselines/golden_v0
	@echo "PREFLIGHT GREEN — ok to merge"

contract: ## regenerate ROS2 contract from contract.py (then commit both)
	python scripts/gen_contract.py
	pytest tests/contract -q

train: gpu-guard ## launch a real run: make train EXP=<overlay> SEED=42
	@test -n "$(EXP)" || (echo "usage: make train EXP=<overlay-name> [SEED=..]" && exit 1)
	$(ISAACLAB) -p scripts/rsl_rl/train.py --task $(TASK) --headless \
	  --experiment $(EXP) --seed $(SEED)
	@echo "REMINDER: append a line to experiments/LOG.md"

train-ref: gpu-guard ## 300-iter reference run for phase gates (queue overnight)
	for s in 42 43 44; do \
	  $(ISAACLAB) -p scripts/rsl_rl/train.py --task $(TASK) --headless \
	    --num_envs 4096 --max_iterations 300 --seed $$s ; \
	done
