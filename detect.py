"""
Simple Roboflow object detection tester.
Sends a local image to your Monkey Detection or Wild Boar Detection model
and prints back what it found.

Usage:
    python detect.py path/to/image.jpg monkey
    python detect.py path/to/image.jpg boar
"""

import sys
import base64
import requests

API_KEY = "rBelYlcd54HgrRJoeoez"

MODEL_URLS = {
    "monkey": "https://detect.roboflow.com/monkey-detection-ve9zl/12",
    "boar": "https://detect.roboflow.com/wild-boar-deterrent/2",
}


def detect(image_path: str, model: str):
    if model not in MODEL_URLS:
        print(f"Unknown model '{model}'. Choose from: {list(MODEL_URLS.keys())}")
        return

    model_url = MODEL_URLS[model]

    # Read and base64-encode the image
    with open(image_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")

    response = requests.post(
        model_url,
        params={"api_key": API_KEY},
        data=img_base64,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    result = response.json()

    if "predictions" not in result:
        print("Error from Roboflow:", result)
        return

    predictions = result["predictions"]
    if not predictions:
        print(f"No {model} detected in {image_path}.")
        return

    print(f"Found {len(predictions)} detection(s) in {image_path}:")
    for p in predictions:
        print(
            f"  - {p['class']} "
            f"(confidence: {p['confidence']:.2f}, "
            f"position: x={p['x']:.0f}, y={p['y']:.0f}, "
            f"size: {p['width']:.0f}x{p['height']:.0f})"
        )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python detect.py <image_path> <monkey|boar>")
        sys.exit(1)

    image_path = sys.argv[1]
    model_choice = sys.argv[2].lower()
    detect(image_path, model_choice)
