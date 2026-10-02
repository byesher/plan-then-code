import sys
import csv
import io


def main():
    text = sys.stdin.read()
    reader = csv.reader(io.StringIO(text))
    for row in reader:
        print(" | ".join(row))


if __name__ == "__main__":
    main()
