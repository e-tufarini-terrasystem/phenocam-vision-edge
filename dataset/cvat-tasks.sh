#!/bin/sh
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cli="$repository_root/dataset/.venv/bin/cvat-cli"
annotation_root="$repository_root/dataset/work/annotation"
profile=phenocam-local

usage() {
    printf '%s\n' \
        "usage: dataset/cvat-tasks.sh profile|list|upload-openimages|upload-phenocam|export TASK_ID NAME|finish TASK_ID NAME ANNOTATOR REVIEWER" \
        "" \
        "profile            save a local personal-access-token profile interactively" \
        "list               list CVAT tasks" \
        "upload-openimages  create the 202-image Open Images task" \
        "upload-phenocam    create the 350-image PhenoCam task" \
        "export ID NAME     export COCO and a full backup for a completed task" \
        "finish ID NAME A R export, audit, and import openimages or phenocam"
}

require_cli() {
    [ -x "$cli" ] || { printf '%s\n' "error: run 'dataset/cvat.sh cli-setup' first" >&2; exit 1; }
}

create_task() {
    name=$1
    bundle=$2
    existing_id=$("$cli" --profile "$profile" task ls --json | \
        "$repository_root/dataset/.venv/bin/python" -c '
import json, sys
name = sys.argv[1]
matches = [str(task["id"]) for task in json.load(sys.stdin) if task["name"] == name]
if len(matches) > 1:
    raise SystemExit("error: duplicate CVAT task names")
print(matches[0] if matches else "")
' "$name")
    if [ -n "$existing_id" ]; then
        printf '%s\n' "task already exists: $existing_id"
        return
    fi
    set -- "$bundle/images/default"/*.jpg
    [ -f "$1" ] || { printf '%s\n' "error: no bundle images in $bundle" >&2; exit 1; }
    "$cli" --profile "$profile" task create "$name" \
        --labels "$bundle/labels.json" \
        --segment_size 50 \
        --annotation_path "$bundle/annotations.coco.zip" \
        --annotation_format "COCO 1.0" \
        local "$@"
}

export_task() {
    task_id=$1
    export_name=$2
    export_dir="$annotation_root/exports"
    mkdir -p "$export_dir"
    history_dir="$export_dir/history/$(date -u '+%Y%m%dT%H%M%SZ')-$$"
    for previous in \
        "$export_dir/$export_name-reviewed.coco.zip" \
        "$export_dir/$export_name-task-backup.zip"
    do
        if [ -f "$previous" ]; then
            mkdir -p "$history_dir"
            mv "$previous" "$history_dir/"
        fi
    done
    "$cli" --profile "$profile" task export-dataset --format "COCO 1.0" "$task_id" "$export_dir/$export_name-reviewed.coco.zip"
    "$cli" --profile "$profile" task backup "$task_id" "$export_dir/$export_name-task-backup.zip"
}

command_name=${1:-}
case "$command_name" in
    profile)
        require_cli
        "$repository_root/dataset/.venv/bin/python" "$repository_root/dataset/cvat-profile.py"
        ;;
    list)
        require_cli
        "$cli" --profile "$profile" task ls --json
        ;;
    upload-openimages)
        require_cli
        create_task "Public dataset — Open Images positive review" "$annotation_root/cvat/openimages-positive"
        ;;
    upload-phenocam)
        require_cli
        create_task "Public dataset — PhenoCam positive annotation" "$annotation_root/cvat/phenocam-positive"
        ;;
    export)
        require_cli
        task_id=${2:-}
        export_name=${3:-}
        [ -n "$task_id" ] && [ -n "$export_name" ] || { usage >&2; exit 2; }
        export_task "$task_id" "$export_name"
        ;;
    finish)
        require_cli
        task_id=${2:-}
        export_name=${3:-}
        annotator=${4:-}
        reviewer=${5:-}
        [ -n "$task_id" ] && [ -n "$export_name" ] && [ -n "$annotator" ] && [ -n "$reviewer" ] || { usage >&2; exit 2; }
        case "$export_name" in
            openimages) bundle_name=openimages-positive ;;
            phenocam) bundle_name=phenocam-positive ;;
            *) printf '%s\n' "error: NAME must be openimages or phenocam" >&2; exit 2 ;;
        esac
        export_task "$task_id" "$export_name"
        import_dir="$annotation_root/imported"
        mkdir -p "$import_dir"
        cd "$repository_root"
        "$repository_root/dataset/.venv/bin/python" -m dataset.builder import-positive-coco \
            --bundle-dir "$annotation_root/cvat/$bundle_name" \
            --export "$annotation_root/exports/$export_name-reviewed.coco.zip" \
            --output "$import_dir/$export_name-positive.csv" \
            --annotator "$annotator" \
            --reviewer "$reviewer"
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
