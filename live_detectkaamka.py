import cv2
import time
import requests
from inference_sdk import InferenceHTTPClient

CAMERA_URL = "http://192.168.1.181:8081/video"
SMS_GATEWAY_URL = "http://192.168.1.181:8080/send-sms"
API_KEY = "rBelYlcd54HgrRJoeoez"

WORKSPACE_NAME = "monus-workspace-ueglv"
WORKFLOW_ID = "wild-boar-target-detection-1783257066502"

CHECK_INTERVAL = 2
ALERT_COOLDOWN = 30
PHONE_NUMBER = "+919955047959"   # <-- put the real number before running for real

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY,
)

last_alert_time = 0


def send_alert(detected_class, confidence):
    message = f"{detected_class} detected near field! (confidence={confidence:.2f})"
    try:
        response = requests.post(
            SMS_GATEWAY_URL,
            json={"phone": PHONE_NUMBER, "message": message},
            timeout=5
        )
        print("[ALERT SENT]", detected_class, "(confidence=%.2f)" % confidence,
              "-> SMS response:", response.status_code, response.text)
    except Exception as e:
        print("[ALERT FAILED] Could not reach SMS gateway:", e)


def run_detection(frame):
    result = client.run_workflow(
        workspace_name=WORKSPACE_NAME,
        workflow_id=WORKFLOW_ID,
        images={"image": frame},
        use_cache=True,
    )
    return result


def extract_all_predictions(result):
    all_preds = []

    def search(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key.endswith("_predictions") and isinstance(value, dict):
                    preds = value.get("predictions", [])
                    if isinstance(preds, list):
                        animal_name = key.replace("_predictions", "").replace("_", " ").title()
                        for p in preds:
                            p = dict(p)
                            p["animal_name"] = animal_name
                            all_preds.append(p)
                search(value)
        elif isinstance(obj, list):
            for item in obj:
                search(item)

    search(result)
    return all_preds


def main():
    global last_alert_time

    cap = cv2.VideoCapture(CAMERA_URL)
    if not cap.isOpened():
        print("Could not open camera stream. Check the CAMERA_URL and WiFi connection.")
        return

    last_check = 0
    print("System started. Watching camera feed for wild boar / monkey detection...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lost connection to camera.")
            break

        cv2.imshow("Live Feed", frame)

        now = time.time()
        if now - last_check >= CHECK_INTERVAL:
            last_check = now
            try:
                result = run_detection(frame)
                predictions = extract_all_predictions(result)

                if predictions:
                    for p in predictions:
                        label = p.get("animal_name", "Unknown")
                        confidence = p.get("confidence", 0)
                        print("[DETECTED]", label, "(confidence=%.2f)" % confidence)

                    if now - last_alert_time >= ALERT_COOLDOWN:
                        top = predictions[0]
                        send_alert(top.get("animal_name", "Unknown"), top.get("confidence", 0))
                        last_alert_time = now
                    else:
                        remaining = int(ALERT_COOLDOWN - (now - last_alert_time))
                        print("(Cooldown active, %ds remaining)" % remaining)
                else:
                    print("No detection.")
            except Exception as e:
                print("Error contacting Roboflow:", e)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()