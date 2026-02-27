.PHONY: install install-dev test lint typecheck gui mobile-deps mobile-apk clean help

PYTHON := python3
POETRY := poetry
PIP := pip

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

install: ## Install momentum (editable) so the 'momentum' command is on PATH
	$(POETRY) install
	@echo ""
	@echo "Done. Run 'momentum --help' to get started."

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
	rm -rf mobile/.buildozer mobile/bin 2>/dev/null || true
	@echo "Cleaned."
