import sys

from pavd.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main(["build_gallery", *sys.argv[1:]]))
