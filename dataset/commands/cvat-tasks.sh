#!/bin/sh
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cli="$repository_root/dataset/.venv/bin/cvat-cli"
annotation_root="$repository_root/dataset/workspace/annotation"
profile=phenocam-local

usage() {
    printf '%s\n' \
        "usage: dataset/commands/cvat-tasks.sh profile|list|upload-open-images|upload-open-images-supplement|upload-phenocam|upload-v3|upload-v3-clean|read-v3|export TASK_ID NAME|finish TASK_ID NAME ANNOTATOR REVIEWER" \
        "" \
        "profile            save a local personal-access-token profile interactively" \
        "list               list CVAT tasks" \
        "upload-open-images  create the 202-image Open Images task" \
        "upload-open-images-supplement create the reduced supplemental review task" \
        "upload-phenocam    create the 350-image PhenoCam task" \
        "upload-v3          create the v3 project and four initial tasks" \
        "upload-v3-clean    create the two replacement internal tasks" \
        "read-v3            read back the v3 project and tasks as JSON" \
        "export ID NAME     export COCO and a full backup for a completed task" \
        "finish ID NAME A R export, audit, and import open-images or phenocam"
}

v3_project_id() {
    "$cli" --profile "$profile" project ls --json | "$repository_root/dataset/.venv/bin/python" -c '
import json, sys
name = "Phenocam privacy detector v3"
matches = [str(project["id"]) for project in json.load(sys.stdin) if project["name"] == name]
if len(matches) > 1: raise SystemExit("error: duplicate CVAT project names")
print(matches[0] if matches else "")
'
}

create_v3_task() {
    name=$1
    bundle=$2
    project_id=$3
    existing_id=$("$cli" --profile "$profile" task ls --json | "$repository_root/dataset/.venv/bin/python" -c '
import json, sys
name, project = sys.argv[1:]
matches = [str(task["id"]) for task in json.load(sys.stdin) if task["name"] == name and str(task["project_id"]) == project]
if len(matches) > 1: raise SystemExit("error: duplicate CVAT task names")
print(matches[0] if matches else "")
' "$name" "$project_id")
    [ -z "$existing_id" ] || { printf '%s\n' "task already exists: $existing_id"; return; }
    set -- "$bundle/images/default"/*
    [ -f "$1" ] || { printf '%s\n' "error: no bundle images in $bundle" >&2; exit 1; }
    "$cli" --profile "$profile" task create "$name" --project_id "$project_id" \
        --segment_size 50 --annotation_path "$bundle/annotations.coco.zip" \
        --annotation_format "COCO 1.0" local "$@"
}

require_cli() {
    [ -x "$cli" ] || { printf '%s\n' "error: run 'dataset/commands/cvat-server.sh cli-setup' first" >&2; exit 1; }
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
        "$repository_root/dataset/.venv/bin/python" "$repository_root/dataset/commands/cvat-profile.py"
        ;;
    list)
        require_cli
        "$cli" --profile "$profile" task ls --json
        ;;
    upload-open-images)
        require_cli
        create_task "Public dataset — Open Images positive review" "$annotation_root/cvat/open-images-positive"
        ;;
    upload-open-images-supplement)
        require_cli
        create_task "Public dataset — Open Images supplemental review" "$annotation_root/cvat/open-images-supplement-positive"
        ;;
    upload-phenocam)
        require_cli
        create_task "Public dataset — PhenoCam positive annotation" "$annotation_root/cvat/phenocam-positive"
        ;;
    upload-v3)
        require_cli
        project_id=$(v3_project_id)
        if [ -z "$project_id" ]; then
            "$cli" --profile "$profile" project create "Phenocam privacy detector v3" --labels "$annotation_root/cvat/v3-public-pilot/labels.json"
            project_id=$(v3_project_id)
        fi
        [ -n "$project_id" ] || { printf '%s\n' "error: v3 project creation failed" >&2; exit 1; }
        create_v3_task "V3 public PhenoCam mining - pilot 200" "$annotation_root/cvat/v3-public-pilot" "$project_id"
        create_v3_task "V3 public dataset label audit - reported 22" "$annotation_root/cvat/v3-public-label-audit" "$project_id"
        create_v3_task "V3 internal operational dev - representative 120" "$annotation_root/cvat/v3-internal-representative" "$project_id"
        create_v3_task "V3 internal operational mining - informative 120" "$annotation_root/cvat/v3-internal-informative" "$project_id"
        ;;
    upload-v3-clean)
        require_cli
        project_id=$(v3_project_id)
        [ -n "$project_id" ] || { printf '%s\n' "error: create the v3 project first" >&2; exit 1; }
        create_v3_task "V3 internal operational dev - representative 120 - clean" "$annotation_root/cvat/v3-internal-representative-clean" "$project_id"
        create_v3_task "V3 internal operational mining - informative 120 - clean" "$annotation_root/cvat/v3-internal-informative-clean" "$project_id"
        ;;
    read-v3)
        require_cli
        "$cli" --profile "$profile" project ls --json
        "$cli" --profile "$profile" task ls --json
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
            open-images) bundle_name=open-images-positive ;;
            open-images-supplement) bundle_name=open-images-supplement-positive ;;
            phenocam) bundle_name=phenocam-positive ;;
            *) printf '%s\n' "error: NAME must be open-images, open-images-supplement, or phenocam" >&2; exit 2 ;;
        esac
        export_task "$task_id" "$export_name"
        import_dir="$annotation_root/imported"
        mkdir -p "$import_dir"
        cd "$repository_root"
        "$repository_root/dataset/.venv/bin/python" -m dataset.builder import-positive-coco \
            --bundle-dir "$annotation_root/cvat/$bundle_name" \
            --export "$annotation_root/exports/$export_name-reviewed.coco.zip" \
            --output "$import_dir/$bundle_name.csv" \
            --annotator "$annotator" \
            --reviewer "$reviewer"
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
