#!/bin/sh
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
environment_python="$repository_root/dataset/.venv/bin/python"

usage() {
    printf '%s\n' \
        "usage: dataset/workflow.sh setup|status|preflight|test|prepare-openimages-review" \
        "" \
        "setup                      create the Python 3.13 environment" \
        "status                     report progress and the next safe action" \
        "preflight                  validate all inputs needed to resume" \
        "test                       run builder tests" \
        "prepare-openimages-review  resume screening and build the review packet"
}

choose_bootstrap_python() {
    if command -v python3.13 >/dev/null 2>&1; then
        command -v python3.13
    elif command -v python3.12 >/dev/null 2>&1; then
        command -v python3.12
    elif command -v python3.11 >/dev/null 2>&1; then
        command -v python3.11
    else
        printf '%s\n' "error: Python 3.11 or newer is required" >&2
        exit 1
    fi
}

require_environment() {
    if [ ! -x "$environment_python" ]; then
        printf '%s\n' "error: run 'dataset/workflow.sh setup' first" >&2
        exit 1
    fi
}

command_name=${1:-}
case "$command_name" in
    setup)
        bootstrap_python=$(choose_bootstrap_python)
        if [ ! -x "$environment_python" ]; then
            "$bootstrap_python" -m venv "$repository_root/dataset/.venv"
        fi
        "$environment_python" -m pip install -r "$repository_root/dataset/requirements.txt"
        ;;
    status)
        require_environment
        cd "$repository_root"
        "$environment_python" -m dataset.builder workflow-status
        ;;
    preflight)
        require_environment
        cd "$repository_root"
        "$environment_python" -m dataset.builder workflow-preflight
        ;;
    test)
        require_environment
        cd "$repository_root"
        "$environment_python" -m unittest discover -s dataset/tests -v
        ;;
    prepare-openimages-review)
        require_environment
        cd "$repository_root"
        "$environment_python" -m dataset.builder prepare-openimages-review
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
