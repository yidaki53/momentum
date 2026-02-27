.PHONY: install install-dev test lint typecheck gui dist build mobile-deps mobile-apk clean help

PYTHON := python3
POETRY := poetry
PIP := pip

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

install: ## Install momentum into the Poetry virtualenv
	$(POETRY) install
	@echo ""
	@echo "Done. Run 'poetry run momentum --help' to get started."

install-dev: ## Install with dev dependencies
	$(POETRY) install --with dev
	$(POETRY) run pre-commit install 2>/dev/null || true
	@echo ""
	@echo "Dev environment ready."

test: ## Run tests
	$(POETRY) run pytest tests/ -v

lint: ## Run ruff linter
	$(POETRY) run ruff check momentum/ tests/

typecheck: ## Run mypy type checking
	$(POETRY) run mypy momentum/

gui: ## Launch the GUI
	$(POETRY) run momentum gui

# ---------------------------------------------------------------------------
# Distribution
# ---------------------------------------------------------------------------

dist: ## Build a standalone binary with PyInstaller
	$(POETRY) run pyinstaller \
		--onefile \
		--name momentum \
		--add-data "ENCOURAGEMENTS.md:." \
		--add-data "IMAGES.md:." \
		--hidden-import momentum \
		--hidden-import PIL._tkinter_finder \
		--clean \
		momentum/cli.py
	@echo ""
	@echo "Built: dist/momentum"
	@echo "Run ./dist/momentum --help to test it."

build: dist ## Alias for dist

# ---------------------------------------------------------------------------
# Mobile (Kivy + Buildozer)
# ---------------------------------------------------------------------------

mobile-deps: ## Install Buildozer and Kivy dependencies for APK builds
	$(PIP) install buildozer kivy cython
	@echo ""
	@echo "You also need Android SDK/NDK build tools."
	@echo "See: https://buildozer.readthedocs.io/en/latest/installation.html"

mobile-apk: ## Build Android APK (requires buildozer + SDK)
	cd mobile && buildozer android debug

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

clean: ## Remove caches and build artefacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ *.spec 2>/dev/null || true
	rm -rf mobile/.buildozer mobile/bin 2>/dev/null || true
	@echo "Cleaned."
