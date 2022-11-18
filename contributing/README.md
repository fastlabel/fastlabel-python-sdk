# Contributing Guideline

## Pull Request Checklist

Before sending your pull requests, make sure you followed this list.

- Read Contributing Guideline
- Run formatter and linter

## How to setup the development environment

### [Recommended] Creating and using virtual environments

It is recommended to use [venv](https://docs.python.org/3/library/venv.html) so that the existing environment is not affected.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install FastLabel Python SDK in develop mode

```bash
pip install -r requirements.txt
pip install -e .
```

If you have installed the FastLabel Python SDK normally, uninstall it and then do a develop setup.

```bash
pip uninstall fastlabel
pip install -r requirements.txt
pip install -e .
```

### Create .env file

When `.env` is placed, it is configured to read environment variables in pytest and vscode. Please refer to the `.env.template` file to create your own.

### Install develop tools

```bash
pip install -r contributing/requirements.txt
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

## Run Formatter and Linter

```bash
make all
```

If the pre-commit hook is enabled, the same process is performed at commit time.

## Test

```bash
pytest tests/
```
