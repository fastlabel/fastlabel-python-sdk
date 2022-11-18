# Contributing Guideline

## Pull Request Checklist

Before sending your pull requests, make sure you followed this list.

- Read Contributing Guideline
- Run formatter and linter

## [Recommended] Creating and using virtual environments

It is recommended to use [venv](https://docs.python.org/3/library/venv.html) so that the existing environment is not affected.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment variables

When `.env` is placed, it is configured to read environment variables in pytest and vscode. Please refer to the `.env.template` file to create your own.

## Formatter and Linter

### Installation

```bash
pip install -r contributing/requirements.txt
```

### Run Formatter and Linter

```bash
make all
```

### Enable pre-commit hook

```bash
pre-commit install
```

Basically, `pre-commit` will only run on the changed files.
But if you execute the bellow command, you can run on all files.

```bash
pre-commit run --all-files
```

## Test

```bash
pytest tests/
```
