#!/bin/sh
# Run single-image inference over the project's input/ directory.
# This script owns batch iteration and output-directory creation.
# Image validation, inference, and output writing remain inside phenocam.

set -u

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd) || { echo "error: project directory cannot be resolved" >&2; exit 1; }
input_dir="$root/input"
output_dir="$root/output"
python="$root/.venv/bin/python"
[ -x "$python" ] || python=$(command -v python3)
status=0
found=0

[ -d "$input_dir" ] || { echo "error: input directory does not exist" >&2; exit 1; }
[ -n "$python" ] && [ -x "$python" ] || { echo "error: Python interpreter does not exist" >&2; exit 1; }
mkdir -p -- "$output_dir" 2>/dev/null || { echo "error: output directory could not be created" >&2; exit 1; }
cd "$root" || { echo "error: project directory cannot be resolved" >&2; exit 1; }

for image in "$input_dir"/*; do
    [ -f "$image" ] || continue
    case "$image" in
        *.[jJ][pP][gG]|*.[jJ][pP][eE][gG]|*.[pP][nN][gG]|*.[wW][eE][bB][pP]|*.[bB][mM][pP]|*.[tT][iI][fF]|*.[tT][iI][fF][fF]) ;;
        *) continue ;;
    esac

    found=1
    name=$(basename -- "$image")
    if ! "$python" -m phenocam --input "$image" --output "$output_dir/$name" --model "$root/models/yolo26n.onnx"; then
        echo "error: inference failed for $name" >&2
        status=1
    fi
done

[ "$found" -eq 1 ] || { echo "error: input directory contains no supported images" >&2; exit 1; }
exit "$status"
