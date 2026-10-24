# display help information
default:
    @just --list


# clean the project
clean:
    rm -rf .venv/


# install the dependencies
install:
    uv sync --all-groups
    uvx prek install


# format with ruff
format:
    uvx ruff format

# lint with ruff
lint:
    uvx ruff check

# Typecheck using ty
typecheck:
    uvx ty check

# Run pre-commit [lint, format]"
pre-commit: lint format
    uvx prek run

# test with pytest
test:
    uv run pytest
