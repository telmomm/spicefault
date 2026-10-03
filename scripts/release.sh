#!/usr/bin/env bash
# Prepare a release: check, set the version, build, commit and tag.
#
#   scripts/release.sh 0.2.0            prepare version 0.2.0
#   scripts/release.sh 0.2.0 --dry-run  everything except the commit and the tag
#
# Publishing is done by GitHub: push the tag and publish a release, and the Publish
# workflow uploads the package to PyPI. See docs/releasing.md.
set -euo pipefail

version="${1:-}"
dry_run="${2:-}"
if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([a-z]+[0-9]+)?$ ]]; then
    echo "usage: scripts/release.sh X.Y.Z [--dry-run]" >&2
    exit 1
fi
cd "$(dirname "$0")/.."
python="${PYTHON:-.venv/bin/python}"

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "the working tree has uncommitted changes: commit or stash them first" >&2
    exit 1
fi
if git rev-parse "v$version" >/dev/null 2>&1; then
    echo "the tag v$version already exists" >&2
    exit 1
fi
if ! grep -q "^## \[$version\]" CHANGELOG.md; then
    echo "CHANGELOG.md has no section '## [$version]': describe the release there first" >&2
    exit 1
fi

echo "== lint and unit tests"
"$python" -m ruff check .
"$python" -m pytest tests/unit -q

echo "== version $version"
today="$(date +%Y-%m-%d)"
"$python" - "$version" "$today" <<'PY'
import re
import sys
from pathlib import Path

version, today = sys.argv[1:]
init = Path("src/spicefault/__init__.py")
text, n = re.subn(r'__version__ = "[^"]+"', f'__version__ = "{version}"', init.read_text())
assert n == 1, "version line not found in src/spicefault/__init__.py"
init.write_text(text)
cff = Path("CITATION.cff")
text = re.sub(r"(?m)^version: .*$", f"version: {version}", cff.read_text())
text = re.sub(r'(?m)^date-released: .*$', f'date-released: "{today}"', text)
cff.write_text(text)
PY

echo "== build"
rm -rf dist
"$python" -m build
"$python" -m twine check dist/*

if [[ "$dry_run" == "--dry-run" ]]; then
    git checkout -- src/spicefault/__init__.py CITATION.cff
    echo "dry run: the package was built in dist/, nothing was committed"
    exit 0
fi

git add src/spicefault/__init__.py CITATION.cff
git commit -m "Release $version"
git tag -a "v$version" -m "spicefault $version"
cat <<MSG

Version $version is committed and tagged. To publish it:

    git push origin main --follow-tags
    gh release create v$version --title "spicefault $version" --notes-from-tag

Publishing the release on GitHub uploads the package to PyPI.
MSG
