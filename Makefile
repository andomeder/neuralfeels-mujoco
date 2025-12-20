# Makefile - NeuralFeels-MuJoCo with mise + uv
# Handles multi-GPU support (Intel Arc XPU, NVIDIA CUDA, AMD ROCm)

export PYTHONPATH=$(shell pwd)

.PHONY: help install install-dev format lint test clean setup-env gpu-info
.PHONY: collect-data train eval demo video docker-build docker-run
.PHONY: setup-mise setup-uv dev-setup quick-test

# Colors
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m

# Detect GPU backend
COMPUTE_BACKEND := cpu
ifeq ($(shell lspci 2>/dev/null | grep -i nvidia | wc -l), 1)
	COMPUTE_BACKEND := nvidia
endif
ifeq ($(shell lspci 2>/dev/null | grep -i "vga.*intel.*arc" | wc -l), 1)
	COMPUTE_BACKEND := intel
endif
ifeq ($(shell lspci 2>/dev/null | grep -i amd | grep -i radeon | wc -l), 1)
	COMPUTE_BACKEND := amd
endif

# Python version
PYTHON_VERSION := 3.12

help: ## Show this help message
	@printf "${BLUE}NeuralFeels-MuJoCo: Visuotactile Perception for Dexterous Manipulation${NC}\n"
	@printf "${YELLOW}Available commands:${NC}\n"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[0;32m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@printf "\n${YELLOW}Current GPU backend:${NC} ${BLUE}$(COMPUTE_BACKEND)${NC}\n"

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

setup-mise: ## Install mise and set up tool versions
	@printf "${BLUE}Setting up mise...${NC}\n"
	@if ! command -v mise &> /dev/null; then \
		curl https://mise.run | sh; \
		echo 'eval "$(mise activate bash)"' >> ~/.bashrc; \
		printf "${YELLOW}Mise installed. Run: source ~/.bashrc${NC}\n"; \
	else \
		printf "${GREEN}✓ Mise already installed: $(mise --version)${NC}\n"; \
	fi
	@mise install
	@mise use python@$(PYTHON_VERSION)
	@printf "${GREEN}✓ Python $(PYTHON_VERSION) activated via mise${NC}\n"

setup-uv: ## Install uv package manager
	@printf "${BLUE}Setting up uv...${NC}\n"
	@if ! command -v uv &> /dev/null; then \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		printf "${YELLOW}uv installed. May need to reload shell${NC}\n"; \
	else \
		printf "${GREEN}✓ uv already installed: $(uv --version)${NC}\n"; \
	fi

install: setup-mise setup-uv ## Install all dependencies (auto-detects GPU)
	@printf "${BLUE}Installing dependencies for $(COMPUTE_BACKEND) backend...${NC}\n"
	@if [ ! -d ".venv" ]; then \
		printf "${YELLOW}Creating virtual environment...${NC}\n"; \
		uv venv .venv; \
	fi
	@uv pip install -e .
ifeq ($(COMPUTE_BACKEND), intel)
	@printf "${BLUE}Installing Intel XPU dependencies (3-step process)...${NC}\n"
	@printf "${YELLOW}Step 1/3: Installing PyTorch with XPU support${NC}\n"
	@uv pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
		--index-url https://download.pytorch.org/whl/xpu
	@printf "${GREEN}✓ Intel XPU dependencies installed${NC}\n"
	@printf "${BLUE}Running sanity test...${NC}\n"
	@uv run python -c "import torch; \
		print(f'PyTorch: {torch.__version__}'); \
		[print(f'[{i}]: {torch.xpu.get_device_properties(i)}') for i in range(torch.xpu.device_count())];" \
		|| printf "${RED}⚠️  XPU test failed - check drivers${NC}\n"
else ifeq ($(COMPUTE_BACKEND), nvidia)
	@printf "${BLUE}Installing NVIDIA CUDA dependencies...${NC}\n"
	@uv pip install -e ".[nvidia]"
else ifeq ($(COMPUTE_BACKEND), amd)
	@printf "${BLUE}Installing AMD ROCm dependencies...${NC}\n"
	@uv pip install -e ".[amd]" --index-url https://download.pytorch.org/whl/rocm6.0
else
	@printf "${YELLOW}No GPU detected, using CPU-only PyTorch${NC}\n"
	@uv pip install torch torchvision torchaudio
endif
	@uv run pre-commit install || true
	@printf "${GREEN}✓ Installation complete!${NC}\n"

install-dev: setup-mise setup-uv ## Install with development extras
	@printf "${BLUE}Installing with development extras...${NC}\n"
	@uv pip install -e ".[dev]"
	@$(MAKE) install
	@printf "${GREEN}✓ Development environment ready!${NC}\n"

lock: setup-uv ## Create lock file for reproducibility
	@printf "${BLUE}Creating lock file...${NC}\n"
	@uv lock
	@printf "${GREEN}✓ Lock file created: uv.lock${NC}\n"

setup-env: ## Set up environment variables
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		printf "${YELLOW}Created .env file. Please edit with your settings.${NC}\n"; \
	else \
		printf "${GREEN}✓ .env file already exists${NC}\n"; \
	fi

# =============================================================================
# DATA & TRAINING
# =============================================================================

collect-data: ## Run interactive data collection
	@printf "${BLUE}Starting data collection...${NC}\n"
	@printf "${YELLOW}Controls:${NC}\n"
	@printf "  W/A/S/D/Q/E: Arm movement\n"
	@printf "  G/H: Gripper close/open\n"
	@printf "  J/K: Torso lift\n"
	@printf "  X: Save episode | Z: Reset\n"
	@uv run python scripts/collect_data.py

train: ## Train neural SDF perception
	@printf "${BLUE}Starting training...${NC}\n"
	@uv run python scripts/train.py
	@printf "${GREEN}✓ Training complete! Check outputs/checkpoints/${NC}\n"

eval: ## Evaluate on test set
	@printf "${BLUE}Running evaluation...${NC}\n"
	@uv run python scripts/eval.py
	@printf "${GREEN}✓ Evaluation complete! Check outputs/metrics/${NC}\n"

demo: ## Run live visualization
	@printf "${BLUE}Starting live demo...${NC}\n"
	@uv run python scripts/demo.py --mode live

video: ## Generate demo video
	@printf "${BLUE}Generating demo video...${NC}\n"
	@uv run python scripts/demo.py --mode video --output outputs/videos/demo.mp4
	@printf "${GREEN}✓ Video saved to outputs/videos/demo.mp4${NC}\n"

# =============================================================================
# CODE QUALITY
# =============================================================================

format: ## Format code with black and ruff
	@printf "${BLUE}Formatting code...${NC}\n"
	@uv run black envs/ perception/ src/ scripts/ tests/
	@uv run ruff check envs/ perception/ src/ scripts/ tests/ --fix
	@printf "${GREEN}✓ Code formatted${NC}\n"

lint: ## Lint code with ruff
	@printf "${BLUE}Linting code...${NC}\n"
	@uv run ruff check envs/ perception/ src/ scripts/ tests/
	@uv run black --check envs/ perception/ src/ scripts/ tests/
	@printf "${GREEN}✓ Linting passed${NC}\n"

test: ## Run test suite
	@printf "${BLUE}Running tests...${NC}\n"
	@uv run pytest tests/ -v --cov=. --cov-report=html --cov-report=term
	@printf "${GREEN}✓ Tests completed! Coverage report in htmlcov/${NC}\n"

pre-commit: ## Run all pre-commit hooks
	@printf "${BLUE}Running pre-commit hooks...${NC}\n"
	@uv run pre-commit run --all-files
	@printf "${GREEN}✓ Pre-commit checks passed${NC}\n"

check: format lint test ## Format, lint, and test (pre-commit prep)
	@printf "${GREEN}✓ All checks passed - ready to commit${NC}\n"

# =============================================================================
# UTILITIES
# =============================================================================

gpu-info: ## Display GPU information
	@printf "${BLUE}GPU Detection Results:${NC}\n"
	@printf "Detected backend: ${GREEN}$(COMPUTE_BACKEND)${NC}\n\n"
	@if command -v lspci >/dev/null 2>&1; then \
		printf "${YELLOW}Available GPUs:${NC}\n"; \
		lspci | grep -i vga || printf "  None detected\n"; \
		printf "\n"; \
	fi
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		printf "${YELLOW}NVIDIA GPU Details:${NC}\n"; \
		nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; \
	fi
	@uv run python -c "import torch; print(f'PyTorch version: {torch.__version__}'); \
		print(f'CUDA available: {torch.cuda.is_available()}'); \
		xpu_available = hasattr(torch, 'xpu') and torch.xpu.is_available() if hasattr(torch, 'xpu') else False; \
		print(f'XPU available: {xpu_available}')" 2>/dev/null || printf "${RED}PyTorch not installed yet${NC}\n"

clean: ## Clean temporary files and caches
	@printf "${BLUE}Cleaning project...${NC}\n"
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} + || true
	@rm -rf .pytest_cache/ htmlcov/ .ruff_cache/
	@printf "${GREEN}✓ Project cleaned${NC}\n"

clean-all: clean ## Deep clean (includes outputs and datasets)
	@printf "${BLUE}Deep cleaning...${NC}\n"
	@rm -rf outputs/checkpoints/* outputs/videos/* outputs/metrics/*
	@rm -rf datasets/*
	@rm -rf uv.lock
	@printf "${GREEN}✓ Deep cleanup complete${NC}\n"

monitor: ## Open WandB dashboard
	@printf "${BLUE}Opening WandB dashboard...${NC}\n"
	@uv run python -c "import webbrowser; webbrowser.open('https://wandb.ai')"

update: setup-uv ## Update all dependencies
	@printf "${BLUE}Updating dependencies...${NC}\n"
	@uv lock --upgrade
	@$(MAKE) install
	@printf "${GREEN}✓ Dependencies updated${NC}\n"

# =============================================================================
# DOCKER
# =============================================================================

docker-build: ## Build Docker image
	@printf "${BLUE}Building Docker image...${NC}\n"
	@docker build --build-arg COMPUTE_BACKEND=$(COMPUTE_BACKEND) -t neuralfeels-mujoco:latest .
	@printf "${GREEN}✓ Docker image built${NC}\n"

docker-run: ## Run container interactively
	@printf "${BLUE}Running Docker container...${NC}\n"
	@docker run -it --rm \
		--gpus all \
		-v $(PWD):/workspace \
		-v ~/.config/wandb:/root/.config/wandb \
		-e WANDB_PROJECT=neuralfeels-mujoco \
		neuralfeels-mujoco:latest bash

# =============================================================================
# DEVELOPMENT
# =============================================================================

dev-setup: ## Complete development environment setup
	@$(MAKE) setup-mise
	@$(MAKE) setup-uv
	@$(MAKE) setup-env
	@$(MAKE) install
	@printf "${GREEN}✓ Development environment ready!${NC}\n\n"
	@printf "${YELLOW}Next steps:${NC}\n"
	@printf "  1. ${GREEN}make collect-data${NC}  - Collect demonstrations\n"
	@printf "  2. ${GREEN}make train${NC}         - Train neural SDF\n"
	@printf "  3. ${GREEN}make eval${NC}          - Evaluate and compute metrics\n"
	@printf "  4. ${GREEN}make demo${NC}          - Run live visualization\n"
	@printf "  5. ${GREEN}make video${NC}         - Generate demo video\n"

quick-test: ## Quick verification that everything works
	@printf "${BLUE}Running quick verification...${NC}\n"
	@uv run python -c "import torch; import mujoco; import gymnasium; print('✓ Core imports successful')"
	@uv run python -c "import open3d; import cv2; print('✓ Visualization imports successful')"
	@printf "${GREEN}✓ Quick test passed!${NC}\n"

verify-setup: ## Verify complete installation
	@printf "${BLUE}Verifying installation...${NC}\n"
	@printf "\n${YELLOW}1. Python version:${NC}\n"
	@python --version
	@printf "\n${YELLOW}2. mise status:${NC}\n"
	@mise list || printf "${RED}mise not configured${NC}\n"
	@printf "\n${YELLOW}3. uv status:${NC}\n"
	@uv --version
	@printf "\n${YELLOW}4. GPU detection:${NC}\n"
	@$(MAKE) gpu-info
	@printf "\n${YELLOW}5. Package verification:${NC}\n"
	@$(MAKE) quick-test
	@printf "\n${GREEN}✓ All systems operational!${NC}\n"

init-project: ## Initialize project structure (run after git clone)
	@printf "${BLUE}Initializing project structure...${NC}\n"
	@mkdir -p configs/{env,perception,train}
	@mkdir -p envs/assets/objects
	@mkdir -p perception
	@mkdir -p src/utils
	@mkdir -p scripts
	@mkdir -p tests
	@mkdir -p datasets
	@mkdir -p outputs/{checkpoints,videos,metrics}
	@mkdir -p odocs/reference_code
	@touch envs/__init__.py perception/__init__.py src/__init__.py src/utils/__init__.py
	@printf "${GREEN}✓ Project structure created${NC}\n"

# =============================================================================
# DATASET MANAGEMENT
# =============================================================================

list-data: ## List collected episodes
	@printf "${YELLOW}Collected Episodes:${NC}\n"
	@ls -1 datasets/ 2>/dev/null | grep episode || printf "  ${RED}No episodes found${NC}\n"
	@printf "\n${YELLOW}Total episodes:${NC} "
	@ls -1 datasets/ 2>/dev/null | grep episode | wc -l

test-dataset: ## Test dataset loading
	@printf "${BLUE}Testing dataset loading...${NC}\n"
	@uv run python -c "\
		import os; \
		import numpy as np; \
		from pathlib import Path; \
		episodes = sorted(Path('datasets').glob('episode_*')); \
		print(f'Found {len(episodes)} episodes'); \
		if episodes: \
			ep = episodes[0]; \
			rgb = sorted((ep / 'rgb').glob('*.png')); \
			print(f'Episode 0: {len(rgb)} frames'); \
			print('✓ Dataset loading works')"

# =============================================================================
# SPECIFIC TESTS
# =============================================================================

test-env: ## Test Allegro Hand environment
	@printf "${BLUE}Testing Allegro Hand environment...${NC}\n"
	@uv run python -c "\
		from envs.allegro_hand_env import AllegroHandEnv; \
		env = AllegroHandEnv(); \
		obs, info = env.reset(); \
		print('✓ Environment initialized'); \
		action = env.action_space.sample(); \
		obs, reward, term, trunc, info = env.step(action); \
		print('✓ Environment step works'); \
		env.close()"

test-tactile: ## Test tactile simulation
	@printf "${BLUE}Testing tactile simulation...${NC}\n"
	@uv run python -c "\
		from envs.tactile_sim import get_tactile_depth; \
		import numpy as np; \
		print('✓ Tactile simulation imports work')"

test-neural-sdf: ## Test neural SDF
	@printf "${BLUE}Testing neural SDF...${NC}\n"
	@uv run python -c "\
		from perception.neural_sdf import NeuralSDF; \
		import torch; \
		model = NeuralSDF(); \
		points = torch.randn(100, 3); \
		sdf = model(points); \
		print(f'✓ Neural SDF forward pass: {sdf.shape}'); \
		print('✓ Neural SDF works')"
