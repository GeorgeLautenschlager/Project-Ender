.PHONY: install test format lint typecheck check pre-commit

install:
	poetry install

test:
	poetry run pytest

format:
	poetry run black .

lint:
	poetry run ruff check .

typecheck:
	poetry run mypy src/

check: format lint typecheck test

pre-commit:
	poetry run pre-commit run --all-files
