import cv2

# Your phone's IP Webcam address + /video (this is the actual video stream endpoint)
camera_url = "http://192.168.1.181:8080/video"

cap = cv2.VideoCapture(camera_url)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Couldn't get frame from camera")
        break

    cv2.imshow("Live Feed", frame)  # shows the video in a window, just for testing

    if cv2.waitKey(1) & 0xFF == ord('q'):  # press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()