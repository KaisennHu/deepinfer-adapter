"""Wait for an actual HTTP health response; does not start or alter a server."""
import argparse
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(args.url, timeout=5) as response:
                if response.status == 200:
                    print(f"Ready: {args.url}")
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    raise SystemExit(f"Readiness deadline exceeded: {args.url}; inspect the worker log")


if __name__ == "__main__":
    main()
