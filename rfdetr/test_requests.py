"""
Simple local tests.

Run API first:
    python run_api.py

Then test:
    python test_requests.py --image path/to/image.jpg
"""

import argparse
import requests


def test_image(path, url):
    with open(path, "rb") as f:
        r = requests.post(
            f"{url}/detect",
            files={"file": f},
            data={"threshold": 0.5, "return_annotated": "true"},
            timeout=120,
        )
    print(r.status_code)
    print(r.text[:2000])


def test_health(url):
    r = requests.get(f"{url}/health", timeout=20)
    print(r.status_code)
    print(r.json())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--image")
    args = parser.parse_args()

    test_health(args.url)

    if args.image:
        test_image(args.image, args.url)
