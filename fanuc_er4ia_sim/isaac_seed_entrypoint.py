"""Set the layout seed inside Isaac's container before loading a script."""
import os
from pathlib import Path
import runpy
import sys


def main():
    if len(sys.argv) < 3:
        raise SystemExit("Usage: isaac_seed_entrypoint.py INTEGER_SEED SCRIPT [ARGS]")
    try:
        seed = int(sys.argv[1])
    except ValueError:
        raise SystemExit("CUBE_LAYOUT_SEED must be an integer") from None
    script = Path(sys.argv[2]).resolve()
    os.environ["CUBE_LAYOUT_SEED"] = str(seed)
    sys.argv = [str(script), *sys.argv[3:]]
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
