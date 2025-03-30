import cv2
import numpy as np
from flask import Flask, Response, render_template, send_from_directory
import threading
import time
import os
from ultralytics import YOLO  # Import the YOLO model

app = Flask(__name__, 
            static_folder='./frontend/build/static',
            template_folder='./frontend/build')

# Global variables
output_frame = None
lock = threading.Lock()
stop_event = threading.Event()
model = None  # Will hold our YOLO model

def load_model():
    """
    Load the YOLOv8 model
    """
    global model
    try:
        # Load a pre-trained YOLOv8 model (this will download it if not present)
        print("Loading YOLOv8 model...")
        model = YOLO('yolov8n.pt')  # Use the nano model for speed, or 's', 'm', 'l', 'x' for larger models
        print("YOLOv8 model loaded successfully")
    except Exception as e:
        print(f"Error loading YOLOv8 model: {e}")
        return False
    return True

def detect_humans(frame):
    """
    Detect humans in a frame using YOLOv8
    """
    global model
    
    if model is None:
        if not load_model():
            # If model loading fails, fall back to a simple frame return
            return frame
    
    # Run YOLOv8 inference on the frame
    results = model(frame)
    
    # Process the results
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        
        for box in boxes:
            # Get box coordinates and class
            x1, y1, x2, y2 = box.xyxy[0]  # Get the xyxy coordinates
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Get class name
            class_name = result.names[cls]
            
            # Only draw bounding boxes for persons (class 0 in COCO dataset)
            if class_name == 'person' and conf > 0.5:  # Confidence threshold
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label with confidence
                label = f'Person: {conf:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return frame

def process_video():
    """
    Process the video file, detect humans, and update the output frame
    """
    global output_frame, lock, stop_event
    
    # Open the video file
    video_path = 'test_video.mp4'
    print(f"Attempting to open video file at: {video_path}")
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print(f"Error: Could not open video file {video_path}")
        # Try with absolute path
        abs_path = os.path.abspath(video_path)
        print(f"Trying with absolute path: {abs_path}")
        video = cv2.VideoCapture(abs_path)
        if not video.isOpened():
            print(f"Error: Still could not open video file with absolute path")
            return
    
    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video loaded successfully: {width}x{height}, {fps} FPS, {frame_count} frames")
    
    # Load the YOLO model
    load_model()
    
    frame_delay = 1 / fps
    
    while not stop_event.is_set():
        success, frame = video.read()
        
        # If the frame was not read successfully, we've reached the end of the video
        if not success:
            print("End of video file. Restarting...")
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning of video
            continue
        
        # Detect humans in the frame
        processed_frame = detect_humans(frame)
        
        # Add timestamp
        timestamp = time.strftime("%A %d %B %Y %I:%M:%S %p", time.localtime())
        cv2.putText(processed_frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Update the output frame
        with lock:
            output_frame = processed_frame.copy()
        
        # Control the frame rate
        time.sleep(frame_delay)
    
    # Release the video when done
    video.release()

def generate():
    """
    Generate frames for the video stream
    """
    global output_frame, lock
    
    frame_count = 0
    while True:
        with lock:
            if output_frame is None:
                print("No frame available yet, waiting...")
                time.sleep(0.1)
                continue
            
            # Encode the frame in JPEG format
            (flag, encoded_image) = cv2.imencode(".jpg", output_frame)
            
            if not flag:
                print("Failed to encode image")
                continue
            
            frame_count += 1
            if frame_count % 30 == 0:  # Log every 30 frames
                print(f"Streaming frame {frame_count}, size: {len(encoded_image)} bytes")
        
        # Yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encoded_image) + b'\r\n')

@app.route("/")
def index():
    """
    Serve the React app
    """
    return render_template("index.html")

@app.route("/api/video_feed")
def video_feed():
    """
    Return the response generated along with the specific media type (MIME type)
    """
    try:
        print("Video feed requested")
        return Response(generate(),
                      mimetype="multipart/x-mixed-replace; boundary=frame")
    except Exception as e:
        print(f"Error in video_feed: {e}")
        return f"Error: {str(e)}", 500

# Serve React static files
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('./frontend/build', path)

@app.route("/test")
def test_page():
    """
    Simple test page to verify Flask is working
    """
    return """
    <html>
    <head>
        <title>Test Page</title>
    </head>
    <body>
        <h1>Test Page</h1>
        <p>If you can see this, Flask is serving HTML correctly.</p>
        <p>Try the <a href="/">main page</a> again.</p>
    </body>
    </html>
    """

if __name__ == "__main__":
    # Start a thread to process the video
    video_thread = threading.Thread(target=process_video)
    video_thread.daemon = True
    video_thread.start()
    
    # Start the Flask app
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True, use_reloader=False)
    
    # Set the stop event when the app is shutting down
    stop_event.set() 