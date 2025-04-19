from flask import Flask, render_template, request, Response
from ultralytics import YOLO
import os
import uuid
import cv2

app = Flask(__name__)

# Load your YOLOv8 model (replace with 'best.pt' if needed)
model = YOLO('yolov8s.pt')

# Directories for uploads and results
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Function to retrieve connected camera names (or IDs)
def get_camera_name(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        return f"Camera {camera_index}"
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'images' not in request.files:
        return 'No images uploaded', 400

    files = request.files.getlist('images')
    if not files:
        return 'No selected files', 400

    result_images = []

    for file in files:
        if file.filename == '':
            continue

        image_filename = f"{uuid.uuid4().hex}.jpg"
        image_path = os.path.join(UPLOAD_FOLDER, image_filename)
        file.save(image_path)

        results = model(image_path)
        result_filename = f"result_{image_filename}"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        results[0].save(filename=result_path)

        result_images.append(result_filename)

    return render_template('result.html', result_images=result_images)

@app.route('/live')
def live():
    # Get the camera name (you can customize this to fetch actual camera info if needed)
    camera_name = get_camera_name(0)  # Default to Camera 0
    return render_template('live.html', camera_name=camera_name)

def generate_frames():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            results = model(frame, verbose=False)
            annotated_frame = results[0].plot()

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    except Exception as e:
        print("Error during video streaming:", e)
    finally:
        cap.release()
        cv2.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)
