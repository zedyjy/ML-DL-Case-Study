#!/bin/bash
# Downloads the RealWorld HAR (Univ. Mannheim, 2016) accelerometer CSV data
# for all 15 subjects and 8 activities into the folder this script sits in.
# Result:  s01/ ... s15/  each with the extracted CSV files
#          zips/          the original zip files (safe to delete afterwards)
cd "$(dirname "$0")" || exit 1
BASE="https://www.uni-mannheim.de/media/Einrichtungen/dws/Files_Research/Projects/sensor/realworld"
ACTS="walking running sitting standing lying climbingup climbingdown jumping"
mkdir -p zips
fail=0
for s in $(seq 1 15); do
  sd=$(printf "s%02d" "$s")
  mkdir -p "$sd"
  for a in $ACTS; do
    z="zips/${sd}_acc_${a}_csv.zip"
    if [ ! -s "$z" ] || ! unzip -tq "$z" >/dev/null 2>&1; then
      echo "Downloading subject $s - $a"
      curl -fL --retry 3 -sS -o "$z" "$BASE/s$s/data/acc_${a}_csv.zip" || { echo "  FAILED: s$s $a"; fail=$((fail+1)); rm -f "$z"; continue; }
    fi
    unzip -oq "$z" -d "$sd"
  done
  # some activities were recorded in several parts and come as zips inside the zip
  for inner in "$sd"/*.zip; do
    [ -e "$inner" ] || continue
    unzip -oq "$inner" -d "$sd" && rm -f "$inner"
  done
done
echo
echo "Done. CSV files: $(find s?? -name '*.csv' | wc -l | tr -d ' ')   failed downloads: $fail"
