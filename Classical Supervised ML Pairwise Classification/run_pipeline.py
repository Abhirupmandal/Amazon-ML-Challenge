"""
Top-level entrypoint script to execute the Amazon ML Challenge 2026 Entity Resolution pipeline.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from pathlib import Path

# Add project root and code package root to sys.path
root_dir = Path(__file__).resolve().parent
pkg_root = root_dir / "code" / "business_entity_resolution"

for p in [str(root_dir), str(pkg_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.pipeline import main

if __name__ == "__main__":
    main()
