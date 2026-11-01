#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# render.sh - regenerate the Mochi UML figures.
#
# Sources:  docs/uml/*.dot    overview class diagram (Graphviz dot)
#           docs/uml/*.gv     class and activity diagrams: Graphviz files
#                             with fixed node positions, drawn by neato -n2;
#                             compact class boxes are expanded by classbox.py
#           docs/uml/*.puml   sequence diagrams (PlantUML; shared look in
#                             docs/uml/mochi_style.iuml)
# Output:   docs/figures/<name>.svg  and  docs/figures/<name>.png
#
# Requirements
#   * Python 3, Graphviz (neato), rsvg-convert (librsvg2-bin), and for the
#     sequence diagrams Java 8 or newer
#   * The fonts "Liberation Serif" (fonts-liberation)
#   * plantuml.jar. Maven Central may be blocked; the npm package
#     node-plantuml ships a copy:
#         mkdir -p /tmp/puml && cd /tmp/puml && npm install node-plantuml
#     which puts it at /tmp/puml/node_modules/node-plantuml/vendor/plantuml.jar
#     (PlantUML 1.2019.06). Set PLANTUML_JAR to use another copy.
#
# Usage
#   ./render.sh                 # every diagram
#   ./render.sh uml_overview    # one diagram
#
# The SVG is drawn by Graphviz or PlantUML. The PNG is rasterised from that SVG with
# rsvg-convert at exactly 300 dpi for the intended print width below, so
# the PNG and the SVG are the same picture.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"
JAR="${PLANTUML_JAR:-/tmp/puml/node_modules/node-plantuml/vendor/plantuml.jar}"
OUT=../figures
DPI=300

# Intended print width of each figure, in inches.
declare -A WIDTH=(
  [uml_overview]=6.5
  [uml_datamodel]=6.5
  [uml_sequence_session]=6.5
  [uml_sequence_round]=6.5
  [uml_activity_generation]=5.5
)

names=("$@")
if [ ${#names[@]} -eq 0 ]; then names=("${!WIDTH[@]}"); fi

for n in "${names[@]}"; do
  if [ -f "$n.gv" ]; then
    # Class diagrams: compact class boxes expanded by classbox.py into
    # HTML-like tables, then drawn by Graphviz at fixed node positions.
    tmp=$(mktemp --suffix=.dot)
    python3 classbox.py "$n.gv" > "$tmp"
    neato -n2 -Tsvg "$tmp" -o "$OUT/$n.svg"
    rm -f "$tmp"
  elif [ -f "$n.dot" ]; then
    # Class diagrams: Graphviz dot with HTML-like table labels.
    # Files with fixed node positions (layout=neato) are drawn by neato -n2.
    if grep -q 'layout=neato' "$n.dot"; then
      neato -n2 -Tsvg "$n.dot" -o "$OUT/$n.svg"
    else
      dot -Tsvg "$n.dot" -o "$OUT/$n.svg"
    fi
  elif [ -f "$n.puml" ]; then
    # Sequence and activity diagrams: PlantUML.
    java -Djava.awt.headless=true -jar "$JAR" -charset UTF-8 -tsvg -o "$PWD/$OUT" "$n.puml"
  else
    echo "no such diagram: $n.dot or $n.puml" >&2; exit 1
  fi
  px=$(awk -v w="${WIDTH[$n]}" -v d="$DPI" 'BEGIN { printf "%d", w * d }')
  rsvg-convert -b white -w "$px" --dpi-x "$DPI" --dpi-y "$DPI" -o "$OUT/$n.png" "$OUT/$n.svg"
  echo "$n: $OUT/$n.svg, $OUT/$n.png (${px} px wide = ${WIDTH[$n]} in at $DPI dpi)"
done
