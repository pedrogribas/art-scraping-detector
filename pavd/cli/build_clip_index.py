import sys

from pavd.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main(["build_clip_index", *sys.argv[1:]]))
