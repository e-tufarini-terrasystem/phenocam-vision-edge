#!/bin/sh
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
python="$repository_root/dataset/.venv/bin/python"

usage() {
    printf '%s\n' \
        "usage: dataset/commands/dataset-finalization.sh COMMAND [FILES]" \
        "commands: prepare, open-open-images, open-phenocam-a, open-phenocam-b, import, resolve, second-round, complete-retry, import-final, accept-single-review, build, build-v3"
}

command_name=${1:-}
review_root="$repository_root/dataset/workspace/annotation/final-negative-review"
case "$command_name" in
    prepare)
        cd "$repository_root"
        "$python" -m dataset.builder.finalization prepare
        ;;
    open-open-images)
        open -a "Google Chrome" "$review_root/open-images-a.html"
        ;;
    open-phenocam-a)
        open -a "Google Chrome" "$review_root/phenocam-a.html"
        ;;
    open-phenocam-b)
        open -a "Google Chrome" "$review_root/phenocam-b.html"
        ;;
    import)
        [ "$#" -eq 4 ] || { usage >&2; exit 2; }
        cd "$repository_root"
        "$python" -m dataset.builder.finalization import-negatives \
            --openimages "$2" \
            --phenocam-first "$3" \
            --phenocam-second "$4"
        ;;
    resolve)
        [ "$#" -eq 4 ] || { usage >&2; exit 2; }
        cd "$repository_root"
        "$python" -m dataset.builder.finalization resolve \
            --openimages "$2" \
            --phenocam-first "$3" \
            --phenocam-second "$4"
        ;;
    second-round)
        [ "$#" -eq 3 ] || { usage >&2; exit 2; }
        cd "$repository_root"
        "$python" -m dataset.builder.finalization second-round \
            --openimages "$2" \
            --phenocam "$3"
        ;;
    import-final)
        [ "$#" -eq 2 ] || { usage >&2; exit 2; }
        cd "$repository_root"
        "$python" -m dataset.builder.finalization import-final \
            --phenocam-second "$2"
        ;;
    complete-retry)
        [ "$#" -eq 2 ] || { usage >&2; exit 2; }
        cd "$repository_root"
        "$python" -m dataset.builder.finalization complete-retry \
            --openimages "$2"
        ;;
    accept-single-review)
        [ "$#" -eq 1 ] || { usage >&2; exit 2; }
        cd "$repository_root"
        "$python" -m dataset.builder.finalization accept-single-review
        ;;
    build)
        [ "$#" -eq 1 ] || { usage >&2; exit 2; }
        cd "$repository_root"
        "$python" -m dataset.builder.finalization build
        ;;
    build-v3)
        [ "$#" -le 2 ] || { usage >&2; exit 2; }
        cd "$repository_root"
        if [ "$#" -eq 2 ]; then
            "$python" -m dataset.builder.finalization build-v3 --destination "$2"
        else
            "$python" -m dataset.builder.finalization build-v3
        fi
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
