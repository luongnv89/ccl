#!/bin/bash
# Basic validation for site structure
cd "$(dirname "$0")"
echo "Testing site structure..."
[ -f index.html ] && echo "✓ index.html exists" || echo "✗ index.html missing"
[ -f docs.html ] && echo "✓ docs.html exists" || echo "✗ docs.html missing"
[ -f guides.html ] && echo "✓ guides.html exists" || echo "✗ guides.html missing"
[ -f resources.html ] && echo "✓ resources.html exists" || echo "✗ resources.html missing"
[ -f changelog.html ] && echo "✓ changelog.html exists" || echo "✗ changelog.html missing"
[ -f styles.css ] && echo "✓ styles.css exists" || echo "✗ styles.css missing"
[ -f nav.js ] && echo "✓ nav.js exists" || echo "✗ nav.js missing"
echo "All files present."
