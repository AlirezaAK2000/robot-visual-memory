"""CLI convenience wrapper. Install package first; accepts the same arguments as rvm import."""

import sys
from rvm.cli import main

if __name__ == "__main__":
    sys.argv.insert(1, "import")
    main()
