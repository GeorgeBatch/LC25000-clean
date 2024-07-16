# Source: https://github.com/xevolesi/pytorch-fcn/tree/master
ifeq ($(shell uname -s),Linux)
    SHELL := /bin/bash
endif
ifeq ($(shell uname -s),Darwin)
    SHELL := /bin/zsh
endif

.EXPORT_ALL_VARIABLES:
PYTHONPATH := ./
TEST_DIR := tests/
LINT_DIR := ./

lint:
	flake8 ${LINT_DIR}

run_tests:
	pytest -svvv ${TEST_DIR}

reset_logs:
	rm -rf logs
	mkdir logs
	rm -rf wandb

# Call this before commit.
pre_push_test: lint run_tests