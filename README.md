# Flare_2025

```
find . -type f -name '*.zip' -print0 \
  | while IFS= read -r -d '' zipfile; do
      dir="$(dirname "$zipfile")"
      unzip -o "$zipfile" -d "$dir"
    done
```