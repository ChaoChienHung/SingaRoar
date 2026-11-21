.PHONY: help install install-backend install-frontend backend frontend dev build lint

PYTHON ?= python3

help:
	@echo "SingaRoar development commands:"
	@echo "  make install           Install backend and frontend dependencies"
	@echo "  make install-backend   Install Python backend dependencies"
	@echo "  make install-frontend  Install Node frontend dependencies"
	@echo "  make backend           Start the Flask backend on http://127.0.0.1:5000"
	@echo "  make frontend          Start the Vite frontend on http://localhost:5173"
	@echo "  make dev               Start backend and frontend together"
	@echo "  make build             Build the frontend for production"
	@echo "  make lint              Run frontend ESLint checks"

install: install-backend install-frontend

install-backend:
	@if command -v uv >/dev/null 2>&1; then \
		cd backend && uv sync; \
	else \
		$(PYTHON) -m pip install -r backend/requirements.txt; \
	fi

install-frontend:
	cd frontend && npm install

backend:
	@if command -v uv >/dev/null 2>&1; then \
		cd backend && uv run python app.py; \
	else \
		$(PYTHON) backend/app.py; \
	fi

frontend:
	cd frontend && npm run dev

dev:
	@trap 'kill 0' INT TERM EXIT; \
	$(MAKE) backend & \
	$(MAKE) frontend & \
	wait

build:
	cd frontend && npm run build

lint:
	cd frontend && npm run lint
