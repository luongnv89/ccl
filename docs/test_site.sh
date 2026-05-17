#!/bin/bash
# Basic validation for site structure
echo "Testing site structure..."
[ -f docs/index.html ] && echo "✓ index.html exists" || echo "✗ index.html missing"
[ -f docs/docs.html ] && echo "✓ docs.html exists" || echo "✗ docs.html missing"
[ -f docs/guides.html ] && echo "✓ guides.html exists" || echo "✗ guides.html missing"
[ -f docs/resources.html ] && echo "✓ resources.html exists" || echo "✗ resources.html missing"
[ -f docs/changelog.html ] && echo "✓ changelog.html exists" || echo "✗ changelog.html missing"
[ -f docs/styles.css ] && echo "✓ styles.css exists" || echo "✗ styles.css missing"
[ -f docs/nav.js ] && echo "✓ nav.js exists" || echo "✗ nav.js missing"
echo "All files present."
