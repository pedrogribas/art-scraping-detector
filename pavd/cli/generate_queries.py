import sys

from pavd.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main(["generate_queries", *sys.argv[1:]]))
