# NOTE: Used on linux, limited support outside of Linux
#
# A simple makefile to help with small tasks related to development of CARL
# These have been configured to only really run short tasks. Longer form tasks
# are usually completed in github actions.

.PHONY: help install-dev check format pre-commit clean build clean-doc clean-build test doc publish

help:
	@echo "Makefile CARL"
	@echo "* install-dev      to install all dev requirements and install pre-commit"
	@echo "* check            to check the source code for issues"
	@echo "* format           to format the code with black and isort"
	@echo "* pre-commit       to run the pre-commit check"
	@echo "* clean            to clean the dist and doc build files"
	@echo "* build            to build a dist"
	@echo "* test             to run the tests"
	@echo "* doc              to generate and view the html files"
	@echo "* publish          to help publish the current branch to pypi"

PYTHON ?= python
CYTHON ?= cython
PYTEST ?= python -m pytest
CTAGS ?= ctags
PIP ?= python -m pip
MAKE ?= make
ISORT ?= isort
PYDOCSTYLE ?= pydocstyle
MYPY ?= mypy
PRECOMMIT ?= pre-commit
RUFF ?= ruff

DIR := ${CURDIR}
DIST := ${CURDIR}/dist
DOCDIR := ${DIR}/docs
INDEX_HTML := file://${DOCDIR}/html/build/index.html

install-dev:
	$(PIP) install -e ".[dev, docs]"
	pre-commit install

install:
	$(PIP) install -e .

check-ruff:
	$(RUFF) check carl test --check || :

check-isort:
	$(ISORT) carl test --check || :

check-pydocstyle:
	$(PYDOCSTYLE) carl || :

check-mypy:
	$(MYPY) carl || :

# pydocstyle does not have easy ignore rules, instead, we include as they are covered
check: check-ruff check-isort check-mypy check-pydocstyle

pre-commit:
	$(PRECOMMIT) run --all-files

format-ruff:
	$(RUFF) format --silent carl test examples
	$(RUFF) check --fix --silent carl test --exit-zero
	$(RUFF) check --fix carl test --exit-zero

format-isort:
	$(ISORT) carl test

format: format-ruff format-isort

test:
	$(PYTEST) --disable-warnings test

cov-report:
	coverage html -d coverage_html

clean-doc:
	$(MAKE) -C ${DOCDIR} clean

clean-build:
	$(PYTHON) setup.py clean
	rm -rf ${DIST}

# Clean up any builds in ./dist as well as doc
clean: clean-doc clean-build

# Build a distribution in ./dist
build:
	$(PYTHON) setup.py sdist

doc:
	$(MAKE) -C ${DOCDIR} docs
	@echo
	@echo "View docs at:"
	@echo ${INDEX_HTML}

# Publish to testpypi
# Will echo the commands to actually publish to be run to publish to actual PyPi
# This is done to prevent accidental publishing but provide the same conveniences
publish: clean-build build
	$(PIP) install twine
	$(PYTHON) -m twine upload --repository testpypi ${DIST}/*
	@echo
	@echo "Test with the following line:"
	@echo "pip install --index-url https://test.pypi.org/simple/ CARL"
	@echo
	@echo "Once you have decided it works, publish to actual pypi with"
	@echo "python -m twine upload dist/*"
