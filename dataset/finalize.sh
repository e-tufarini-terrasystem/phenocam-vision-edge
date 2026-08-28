#!/bin/sh
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
python="$repository_root/dataset/.venv/bin/python"

usage() {
    printf '%s\n' \
        "usage: dataset/finalize.sh prepare|open-openimages|open-phenocam-a|open-phenocam-b|import OPENIMAGES.csv PHENOCAM_A.csv PHENOCAM_B.csv"
}

command_name=${1:-}
review_root="$repository_root/dataset/work/annotation/final-negative-review"
case "$command_name" in
    prepare)
        cd "$repository_root"
        "$python" -m dataset.builder.finalization prepare
        ;;
    open-openimages)
        open -a "Google Chrome" "$review_root/openimages-a.html"
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
    *)
        usage >&2
        exit 2
        ;;
esac
