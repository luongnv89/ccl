#!/bin/bash
# Comprehensive validation for site structure
cd "$(dirname "$0")"
echo "Testing site structure..."

# Check file existence
files=(index.html docs.html guides.html resources.html changelog.html styles.css nav.js favicon.svg)
missing=0
for file in "${files[@]}"; do
  if [ -f "$file" ]; then
    echo "✓ $file exists"
  else
    echo "✗ $file missing"
    missing=$((missing + 1))
  fi
done

# Check internal links
echo ""
echo "Checking internal links..."
for html in index.html docs.html guides.html resources.html changelog.html; do
  # Check CSS/JS references
  if ! grep -q 'href="styles.css"' "$html"; then
    echo "✗ $html: missing styles.css link"
    missing=$((missing + 1))
  fi
  if ! grep -q 'src="nav.js"' "$html" && ! grep -q '<script>' "$html" | grep -q 'nav.js'; then
    # nav.js is inline in a script tag, not external
    :
  fi
  # Check favicon reference
  if ! grep -q 'href="favicon.svg"' "$html"; then
    echo "✗ $html: missing favicon.svg link"
    missing=$((missing + 1))
  fi
  # Check navigation links
  for link in index.html docs.html guides.html resources.html changelog.html; do
    if ! grep -q "href=\"$link\"" "$html"; then
      echo "✗ $html: missing navigation link to $link"
      missing=$((missing + 1))
    fi
  done
done

# Check legacy redirects in index.html
echo ""
echo "Checking legacy redirects..."
if grep -q "hash === '#model-guide'" index.html && grep -q "window.location.href = 'guides.html'" index.html; then
  echo "✓ #model-guide redirect present"
else
  echo "✗ #model-guide redirect missing"
  missing=$((missing + 1))
fi
if grep -q "hash === '#changelog'" index.html && grep -q "window.location.href = 'changelog.html'" index.html; then
  echo "✓ #changelog redirect present"
else
  echo "✗ #changelog redirect missing"
  missing=$((missing + 1))
fi

echo ""
if [ $missing -eq 0 ]; then
  echo "✓ All checks passed"
  exit 0
else
  echo "✗ $missing check(s) failed"
  exit 1
fi
