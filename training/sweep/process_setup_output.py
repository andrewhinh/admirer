import argparse


def process(url):
    parts = url.split("/")
    print(parts[3], parts[-1])  # Entity and Sweep ID


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--url",
        help="The project to log the sweep results to.",
    )
    args = parser.parse_args()

    process(args.url)


if __name__ == "__main__":
    main()
