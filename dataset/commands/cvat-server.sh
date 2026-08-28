#!/bin/sh
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cvat_root="$repository_root/dataset/workspace/tools/cvat-v2.71.0"
version=v2.71.0

usage() {
    printf '%s\n' \
        "usage: dataset/commands/cvat-server.sh setup|start|stop|status|create-superuser|open|cli-setup" \
        "" \
        "setup             clone the pinned CVAT Community release" \
        "start             start the local Docker Compose services" \
        "stop              stop services while preserving data" \
        "status            show service state" \
        "create-superuser  interactively create the local administrator" \
        "open              open the local UI in Google Chrome" \
        "cli-setup         install the matching CVAT CLI in dataset/.venv"
}

require_checkout() {
    if [ ! -f "$cvat_root/docker-compose.yml" ]; then
        printf '%s\n' "error: run 'dataset/commands/cvat-server.sh setup' first" >&2
        exit 1
    fi
}

command_name=${1:-}
case "$command_name" in
    setup)
        if [ ! -d "$cvat_root/.git" ]; then
            git clone --depth 1 --branch "$version" https://github.com/cvat-ai/cvat.git "$cvat_root"
        fi
        actual_version=$(git -C "$cvat_root" describe --tags --exact-match)
        [ "$actual_version" = "$version" ] || { printf '%s\n' "error: unexpected CVAT checkout: $actual_version" >&2; exit 1; }
        ;;
    start)
        require_checkout
        docker info >/dev/null
        (cd "$cvat_root" && CVAT_VERSION="$version" docker compose up -d)
        ;;
    stop)
        require_checkout
        (cd "$cvat_root" && docker compose stop)
        ;;
    status)
        require_checkout
        (cd "$cvat_root" && docker compose ps)
        ;;
    create-superuser)
        require_checkout
        docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
        ;;
    open)
        open -a "Google Chrome" http://localhost:8080
        ;;
    cli-setup)
        [ -x "$repository_root/dataset/.venv/bin/python" ] || { printf '%s\n' "error: run 'dataset/commands/dataset-builder.sh setup' first" >&2; exit 1; }
        "$repository_root/dataset/.venv/bin/python" -m pip install -r "$repository_root/dataset/requirements-cvat.txt"
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
