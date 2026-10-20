#!/bin/bash
# Refresh the generated parts of the submission folder from the repository.
#   scripts/refresh_submission.sh [destination]      (default: ../submission)
# Commit your changes first: the code zip is built from HEAD, not the working tree.
set -euo pipefail
cd "$(dirname "$0")/.."
DEST="${1:-../submission}"
mkdir -p "$DEST"/{1_report,2_code,3_screenshots}

if [ -n "$(git status --porcelain)" ]; then
  echo "WARNING: uncommitted changes exist; the zip contains only what is committed." >&2
fi

cp report/Heart_Disease_MLOps_Report.pdf report/Heart_Disease_MLOps_Report.docx "$DEST/1_report/"
rsync -a --delete --exclude '.gitkeep' screenshots/ "$DEST/3_screenshots/"
git archive --format=zip --prefix=heart-disease-mlops/ -o "$DEST/2_code/heart-disease-mlops.zip" HEAD

echo "Submission folder refreshed at $(cd "$DEST" && pwd)"
echo "  code zip from commit $(git rev-parse --short HEAD) on branch $(git branch --show-current)"
ls -la "$DEST"/1_report "$DEST"/2_code
echo "Screenshots: $(ls "$DEST"/3_screenshots | wc -l | tr -d ' ') files"
