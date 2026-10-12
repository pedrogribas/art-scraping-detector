import sys

from pavd.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main(["run_all", *sys.argv[1:]]))
