#!/bin/sh
# Process each image in input/ separately, writing annotations to output/.
# Pass a same-stem .meta file when it is regular and not a symlink.

set -u

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd) || { echo "error: project directory cannot be resolved" >&2; exit 1; }

# Settings: paths may be absolute or relative to the project root.
model="$root/models/yolo26n-phenocam.onnx"
input_dir="$root/input"
output_dir="$root/output"
confidence=0.47  # 0..1; higher thresholds also discard more true detections.
threads=${YOLO_NUM_THREADS:-4}  # 1..4

python="$root/.venv/bin/python"
[ -x "$python" ] || python=$(command -v python3)
status=0
found=0

case "$threads" in
    1|2|3|4) export YOLO_NUM_THREADS="$threads" ;;
    *) echo "error: threads must be an integer from 1 to 4" >&2; exit 1 ;;
esac
cd "$root" || { echo "error: project directory cannot be resolved" >&2; exit 1; }
[ -d "$input_dir" ] || { echo "error: input directory does not exist" >&2; exit 1; }
[ -n "$python" ] && [ -x "$python" ] || { echo "error: Python interpreter does not exist" >&2; exit 1; }
mkdir -p -- "$output_dir" 2>/dev/null || { echo "error: output directory could not be created" >&2; exit 1; }

# The CLI has no confidence option; override its filter only in this child process.
runner='
import sys
from functools import partial

try:
    confidence = float(sys.argv.pop(1))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError
except (ValueError, IndexError):
    print("error: confidence must be a number from 0 to 1", file=sys.stderr)
    raise SystemExit(1)

from phenocam.inference import pipeline
from phenocam.__main__ import main

pipeline.normalize_rows = partial(
    pipeline.normalize_rows, confidence_threshold=confidence,
)
raise SystemExit(main())
'

for image in "$input_dir"/*; do
    [ -f "$image" ] || continue
    case "$image" in
        *.[jJ][pP][gG]|*.[jJ][pP][eE][gG]|*.[pP][nN][gG]|*.[wW][eE][bB][pP]|*.[bB][mM][pP]|*.[tT][iI][fF]|*.[tT][iI][fF][fF]) ;;
        *) continue ;;
    esac

    found=1
    name=$(basename -- "$image")
    extension=${name##*.}
    stem=${name%.*}
    annotated="$output_dir/${stem}_annotated.${extension}"
    metadata="$input_dir/${stem}.meta"
    set -- "$python" -c "$runner" "$confidence" --input "$image" --model "$model" --annotated-output "$annotated"
    if [ -f "$metadata" ] && [ ! -L "$metadata" ]; then
        set -- "$@" --meta "$metadata"
    fi
    if ! "$@"; then
        echo "error: inference failed for $name" >&2
        status=1
    fi
done

[ "$found" -eq 1 ] || { echo "error: input directory contains no supported images" >&2; exit 1; }
exit "$status"
