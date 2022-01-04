# Contributing Guideline

## Pull Request Checklist

Before sending your pull requests, make sure you followed this list.

- Read Contributing Guideline
- Run formatter and linter

## Formatter and Linter

### Installation

```bash
$ pip install -r contributing/requirements.txt
```

### Run Formatter and Linter

```bash
$ make all
```

### Enable pre-commit hook

```bash
$ pre-commit install
```

Basically, `pre-commit` will only run on the changed files.
But if you execute the bellow command, you can run on all files.

```bash
$ pre-commit run --all-files
```
