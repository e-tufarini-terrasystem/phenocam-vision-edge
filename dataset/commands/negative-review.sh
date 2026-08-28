#!/bin/sh
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
review_root="$repository_root/dataset/workspace/annotation/negative-review"
environment_python="$repository_root/dataset/.venv/bin/python"

usage() {
    printf '%s\n' \
        "usage: dataset/commands/negative-review.sh open-open-images|open-phenocam-a|open-phenocam-b|import OPEN_IMAGES_CSV PHENOCAM_A_CSV PHENOCAM_B_CSV"
}

open_review() {
    path=$1
    [ -f "$path" ] || { printf '%s\n' "error: run 'dataset/commands/dataset-builder.sh annotation-bundles' first" >&2; exit 1; }
    open -a "Google Chrome" "$path"
}

command_name=${1:-}
case "$command_name" in
    open-open-images)
        open_review "$review_root/open-images-round-a.html"
        ;;
    open-phenocam-a)
        open_review "$review_root/phenocam-round-a.html"
        ;;
    open-phenocam-b)
        open_review "$review_root/phenocam-round-b.html"
        ;;
    import)
        openimages=${2:-}
        phenocam_a=${3:-}
        phenocam_b=${4:-}
        [ -n "$openimages" ] && [ -n "$phenocam_a" ] && [ -n "$phenocam_b" ] || { usage >&2; exit 2; }
        [ -x "$environment_python" ] || { printf '%s\n' "error: run 'dataset/commands/dataset-builder.sh setup' first" >&2; exit 1; }
        output_dir="$repository_root/dataset/workspace/annotation/imported"
        mkdir -p "$output_dir"
        cd "$repository_root"
        "$environment_python" -m dataset.builder import-negative-reviews \
            --openimages "$openimages" \
            --phenocam-first "$phenocam_a" \
            --phenocam-second "$phenocam_b" \
            --output "$output_dir/negative-reviews.csv"
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
