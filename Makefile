.PHONY: install dev run lint format test typecheck refresh

install:
	pip install -e .

dev:
	pip install -e .[dev]

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

lint:
	ruff check app tests

format:
	ruff format app tests && black app tests

 typecheck:
	mypy app

test:
	pytest -s

refresh:
	python -m app.cli refresh
