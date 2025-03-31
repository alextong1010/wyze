import cv2
import numpy as np
from flask import Flask, Response, render_template, send_from_directory, jsonify, request
from flask_socketio import SocketIO, emit, disconnect
import threading
import time
import os
from ultralytics import YOLO
import argparse
import base64

app = Flask(__name__,
            static_folder='./frontend/build/static',
            template_folder='./frontend/build')
app.config['SECRET_KEY'] = 'key'
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*")

# --- Configuration ---
VIDEO_FOLDER = 'videos'
AVAILABLE_VIDEOS = {
    "Video 1": os.path.join(VIDEO_FOLDER, 'test_video.mp4'),
    "Video 2": os.path.join(VIDEO_FOLDER, 'test_video2.mp4'),
    # Add more videos here as needed
}
# Use the first video name as the default key
DEFAULT_VIDEO_KEY = next(iter(AVAILABLE_VIDEOS))
# Get the actual path for the default video
DEFAULT_VIDEO_PATH = AVAILABLE_VIDEOS[DEFAULT_VIDEO_KEY]


# --- Global variables ---
lock = threading.Lock()
stop_event = threading.Event()
switch_video_event = threading.Event()
play_event = threading.Event()
current_video_path = DEFAULT_VIDEO_PATH
next_video_path = None
model = None
video_processing_active = False

# --- Add TV Detection State Variables ---
tv_detection_frames = 10       # Number of frames to check for TV presence at video start
tv_confidence_threshold = 0.65  # << INCREASED THRESHOLD (e.g., to 0.4 or 0.5)
tv_detection_classes = ['tv', 'television', 'monitor', 'screen', 'display']  # Extended class list
debug_mode = False             # Disable debugging for TV detection issues by default

# --- Add Smoothing State Variables ---
stable_lighting_level = 0       # The level currently reported to the frontend
pending_lighting_state = 0      # The state detected in the most recent frames (0 or 8)
frames_in_pending_state = 0     # Counter for consecutive frames in the pending state
STATE_CHANGE_THRESHOLD = 3      # How many frames needed to confirm a state change (adjust as needed)
# --- End Smoothing State Variables ---

# --- Helper Functions ---
def get_video_path(video_name):
    """Gets the full path for a given video name."""
    return AVAILABLE_VIDEOS.get(video_name)

# --- YOLO Model Loading & Detection ---
def load_model():
    """Loads the YOLOv8 model."""
    global model
    if model is None:
        try:
            print("Loading YOLOv8 model...")
            # Use the nano model for speed, or 's', 'm', 'l', 'x'
            model = YOLO('yolov8n.pt')
            print("YOLOv8 model loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading YOLOv8 model: {e}")
            return False
    return True # Already loaded

def detect_objects(frame):
    """Detects objects (specifically persons) in a frame using YOLOv8 and updates lighting level."""
    global model
    current_lighting_level = 0 # Default to 0 for each frame check
    if model is None:
        print("Model not loaded, attempting to load...")
        if not load_model():
            # Draw error message on frame if model fails to load
            cv2.putText(frame, "Error: YOLO Model Load Failed", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # Ensure lighting level is off if model fails
            return frame, current_lighting_level # Return frame and level 0

    # Run YOLOv8 inference
    results = model(frame, verbose=False) # Set verbose=False to reduce console spam

    high_confidence_person_detected = False
    # Process results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            try:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = result.names[cls]

                # Draw bounding box only for persons with confidence > 0.5
                if class_name == 'person':
                    if conf > 0.5:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f'Person: {conf:.2f}'
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    # Check for high confidence to trigger light
                    if conf > 0.5:
                        high_confidence_person_detected = True

            except Exception as e:
                print(f"Error processing detection box: {e}")
                continue # Skip this box if there's an error

    # Update lighting level based on detection
    current_lighting_level = 8 if high_confidence_person_detected else 0
    # No need to lock here as level is local to this frame processing pass

    return frame, current_lighting_level # Return both

def detect_tv_status(frame):
    """Detects if a TV is present in the frame and determines if it's on or off."""
    global model
    tv_detected = False
    tv_status = "off"  # Default status
    best_tv_confidence = 0.0 # Track the highest confidence *for a TV class*
    tv_box = None # Store the box coordinates of the best TV detection
    best_tv_class_name = "" # Store the class name of the best TV detection


    if model is None:
        print("Model not loaded, attempting to load...")
        if not load_model():
            cv2.putText(frame, "Error: YOLO Model Load Failed", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # Return default values if model fails
            return frame, tv_detected, tv_status, best_tv_confidence, tv_box

    # Run YOLOv8 inference
    results = model(frame, verbose=False)

    # Process results - Find the best *TV class* detection
    for result in results:
        boxes = result.boxes
        for box in boxes:
            try:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = result.names[cls]

                # Check if it's a TV class AND meets the threshold AND is better than the last TV found
                if class_name in tv_detection_classes and conf > tv_confidence_threshold:
                    if conf > best_tv_confidence:
                        tv_detected = True # We found at least one valid TV
                        best_tv_confidence = conf
                        tv_box = (x1, y1, x2, y2) # Store the box of this best detection
                        best_tv_class_name = class_name # Store its name

                # Optional: Draw boxes for other high-confidence objects (like persons) for context
                # elif class_name == 'person' and conf > 0.5:
                #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1) # Thinner box maybe
                #     label = f'Person: {conf:.2f}'
                #     cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


            except Exception as e:
                print(f"Error processing detection box: {e}")
                continue  # Skip this box if there's an error

    # If a TV was detected (i.e., tv_detected is True), analyze its state and draw its box
    if tv_detected and tv_box:
        x1, y1, x2, y2 = tv_box
        # Draw the box for the best TV detection
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2) # Changed color for TV box

        # Extract the TV region to analyze if it's on or off
        tv_region = frame[y1:y2, x1:x2]

        if tv_region.size > 0:  # Make sure region is valid
            # Enhanced TV on/off detection
            gray_tv = cv2.cvtColor(tv_region, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray_tv)
            std_brightness = np.std(gray_tv)
            color_variance = np.mean([np.std(tv_region[:,:,0]),
                                     np.std(tv_region[:,:,1]),
                                     np.std(tv_region[:,:,2])])

            # Criteria for TV being ON
            if (std_brightness > 15 and mean_brightness > 25) or color_variance > 12:
                tv_status = "on"
                label = f'{best_tv_class_name}: {best_tv_confidence:.2f} (ON)' # Use stored best TV info
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                tv_status = "off" # Explicitly set to off
                label = f'{best_tv_class_name}: {best_tv_confidence:.2f} (OFF)' # Use stored best TV info
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2) # Match box color
        else:
             tv_status = "off" # Invalid region, assume off

    # Debug info only shown if debug_mode is True and no TV was detected above threshold
    if debug_mode and not tv_detected:
        detected_classes_debug = []
        for result in results:
            for box in result.boxes:
                try: # Add try-except here too
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = result.names[cls]
                    detected_classes_debug.append(f"{class_name}:{conf:.2f}")
                except Exception as e:
                    print(f"Error getting debug class info: {e}")

        if detected_classes_debug:
            debug_text = f"Detected: {', '.join(detected_classes_debug)}"
            cv2.putText(frame, debug_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Return the confidence of the best *TV* detection found
    return frame, tv_detected, tv_status, best_tv_confidence, tv_box


# --- Video Processing Thread ---
def process_video(target_fps=None):
    """Processes video frames, detects objects, applies smoothing, and emits data."""
    global current_video_path, next_video_path, video_processing_active, model
    # Bring smoothing globals into scope
    global stable_lighting_level, pending_lighting_state, frames_in_pending_state

    print("Video processing thread started.")
    video_processing_active = True
    last_frame_time = time.time()
    cap = None

    # --- TV control variables (scoped within the thread) ---
    tv_detected_in_scene = False # Does the current video contain a TV?
    tv_current_state = "off"     # What state do we *think* the TV is in? (off/on)
    # --- End TV control variables ---

    if not load_model():
         print("Initial model load failed. Thread exiting.")
         socketio.emit('video_status', {'status': 'error', 'message': 'Failed to load ML model.'})
         video_processing_active = False
         return # Cannot proceed without model

    while not stop_event.is_set():
        with lock:
            video_path = current_video_path
            video_name = os.path.basename(video_path)

        # --- Video Capture Initialization/Reinitialization ---
        if cap is None or switch_video_event.is_set():
            if cap:
                cap.release()
                print(f"Released previous video capture.")
            if switch_video_event.is_set():
                with lock:
                    video_path = next_video_path if next_video_path else video_path
                    current_video_path = video_path
                    video_name = os.path.basename(video_path)
                    next_video_path = None
                # --- Reset Smoothing State on Switch ---
                print("Resetting lighting state for video switch.")
                stable_lighting_level = 0
                pending_lighting_state = 0
                frames_in_pending_state = 0
                # --- Reset TV State on Switch ---
                tv_detected_in_scene = False
                tv_current_state = "off"
                socketio.emit('tv_status_update', {'status': 'unknown'}) # Reset frontend TV state
                # --- End Reset ---
                switch_video_event.clear()
                play_event.clear()
                print(f"Switching to video: {video_path}")
                socketio.emit('video_status', {'status': 'switching', 'video': video_name})

            print(f"Attempting to open video: {video_path}")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video file {video_path}")
                socketio.emit('video_status', {'status': 'error', 'message': f'Could not open {video_name}', 'video': video_name})
                cap = None # Ensure we retry or stop
                time.sleep(2) # Wait before retrying or exiting loop
                continue # Go back to start of while loop

            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_delay = 1.0 / (target_fps if target_fps else original_fps) if original_fps > 0 else 0.04 # Default ~25fps
            print(f"Video '{video_name}' opened. Target FPS: {1/frame_delay:.2f} (Original: {original_fps:.2f})")

            # --- Signal Ready and Wait for Play ---
            print(f"Video '{video_name}' ready. Waiting for play signal...")
            socketio.emit('video_status', {'status': 'ready', 'video': video_name})
            while not play_event.is_set() and not stop_event.is_set() and not switch_video_event.is_set():
                 play_event.wait(timeout=0.5)
            if stop_event.is_set() or switch_video_event.is_set():
                 print("Stop or Switch signaled while waiting to play.")
                 continue
            if play_event.is_set():
                 print(f"Play signal received for '{video_name}'. Starting stream.")
                 socketio.emit('video_status', {'status': 'playing', 'video': video_name})
            else:
                 continue

            # --- Initial TV Detection Check ---
            print(f"Checking first {tv_detection_frames} frames for TV presence...")
            tv_check_frames = 0
            tv_detection_count = 0
            tv_best_confidence = 0.0
            initial_tv_state = "off" # Track initial detected state

            # Sample several frames to look for TVs
            while tv_check_frames < tv_detection_frames:
                ret, check_frame = cap.read()
                if not ret:
                    print("Warning: Could not read frame during initial TV check.")
                    break # Stop check if video ends early

                try:
                    # Use detect_tv_status to check for TV
                    _, tv_detected, detected_status, tv_conf, _ = detect_tv_status(check_frame.copy()) # Use copy
                    if tv_detected:
                        tv_detection_count += 1
                        tv_best_confidence = max(tv_best_confidence, tv_conf)
                        # If this is the first detection, note its state
                        if tv_detection_count == 1:
                            initial_tv_state = detected_status
                        # print(f"DEBUG: TV detected in frame {tv_check_frames+1} with confidence {tv_conf:.2f}, status: {detected_status}")
                except Exception as e:
                    print(f"Error processing frame {tv_check_frames+1} during TV check: {e}")

                tv_check_frames += 1

                # Optimization: If we've found a TV confidently, maybe stop early?
                # if tv_detection_count >= 3 and tv_best_confidence > 0.3:
                #    print("Confident TV detection, stopping initial check early.")
                #    break

            # Reset video to start for actual playback
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Determine if scene has a TV based on the check
            # Require at least 2 detections to be more certain
            if tv_detection_count >= 2:
                tv_detected_in_scene = True
                # Set the initial *controlled* state based on the first detected frame's status
                tv_current_state = initial_tv_state
                print(f"TV detected in scene ({tv_detection_count}/{tv_check_frames} frames, best conf: {tv_best_confidence:.2f}). Initial state: {tv_current_state.upper()}")
                # Emit initial state to frontend
                socketio.emit('tv_status_update', {'status': tv_current_state})
            else:
                tv_detected_in_scene = False
                tv_current_state = "off" # Default if no TV found
                print("No persistent TV detected in initial frames.")
                socketio.emit('tv_status_update', {'status': 'not_detected'}) # Inform frontend
            # --- End Initial TV Detection Check ---


        # --- Frame Reading and Processing Loop (Only if playing) ---
        if cap and cap.isOpened() and play_event.is_set():
            ret, frame = cap.read()

            if not ret:
                print(f"End of video '{video_name}' or read error.")
                cap.release()
                cap = None
                play_event.clear()
                # --- Reset Smoothing State on Stop/End ---
                stable_lighting_level = 0
                pending_lighting_state = 0
                frames_in_pending_state = 0
                # --- Reset TV State on Stop/End ---
                tv_detected_in_scene = False
                tv_current_state = "off"
                socketio.emit('tv_status_update', {'status': 'unknown'}) # Reset frontend TV state
                # --- End Reset ---
                socketio.emit('video_status', {'status': 'stopped', 'message': 'End of video.', 'video': video_name})
                continue

            # --- Object Detection (gets raw level for this frame) ---
            try:
                # Detect persons first to get lighting level
                processed_frame, detected_level = detect_objects(frame.copy()) # Use copy for safety
                person_detected = (detected_level > 0) # True if lighting level is 8

                # --- TV Detection and Control Logic (only if TV was found initially) ---
                if tv_detected_in_scene:
                    # Now check the TV status in the *current* frame
                    # We use the 'processed_frame' which might already have person boxes drawn
                    tv_frame, tv_visible_now, tv_actual_status, _, _ = detect_tv_status(processed_frame.copy())
                    processed_frame = tv_frame # Update frame with TV box/status drawing

                    # Only attempt control if the TV is visible in this specific frame
                    if tv_visible_now:
                        # Scenario 1: Person detected, but our state is OFF -> Turn ON
                        if person_detected and tv_current_state == "off":
                            tv_current_state = "on"
                            print(f"TV Control: Person detected - Simulating TV ON (Emitting status: {tv_current_state})")
                            socketio.emit('tv_status_update', {'status': tv_current_state})
                            # Add visual indicator (optional)
                            cv2.putText(processed_frame, "TV ON (Simulated)", (frame.shape[1] - 300, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                        # Scenario 2: No person detected, but our state is ON -> Turn OFF
                        elif not person_detected and tv_current_state == "on":
                            tv_current_state = "off"
                            print(f"TV Control: No person detected - Simulating TV OFF (Emitting status: {tv_current_state})")
                            socketio.emit('tv_status_update', {'status': tv_current_state})
                            # Add visual indicator (optional)
                            cv2.putText(processed_frame, "TV OFF (Simulated)", (frame.shape[1] - 300, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)
                        # Optional: Log if actual detected state differs from controlled state
                        # elif tv_actual_status != tv_current_state:
                        #    print(f"Debug: TV actual state '{tv_actual_status}' differs from controlled state '{tv_current_state}'")

                    # else: # TV not visible in this frame
                        # print("DEBUG: TV not visible in current frame, skipping control logic.")
                        # pass # Keep tv_current_state as is if TV is temporarily occluded

                # --- End TV Detection and Control Logic ---

            except Exception as e:
                print(f"Error during object/TV detection or control: {e}")
                # Fallback: use original frame, assume no detection
                processed_frame = frame
                detected_level = 0
                # Optionally reset TV state or log error? For now, just continue.

            # --- Lighting Level Smoothing Logic ---
            with lock: # Protect access to shared smoothing variables
                if detected_level == pending_lighting_state:
                    frames_in_pending_state += 1
                else:
                    pending_lighting_state = detected_level
                    frames_in_pending_state = 1

                if frames_in_pending_state >= STATE_CHANGE_THRESHOLD:
                    if stable_lighting_level != pending_lighting_state:
                        print(f"Lighting state changing: {stable_lighting_level} -> {pending_lighting_state} (threshold met)")
                        stable_lighting_level = pending_lighting_state
                        # Emit lighting level change *here* if you only want to emit on change
                        # socketio.emit('lighting_change', {'level': stable_lighting_level})

                level_to_emit = stable_lighting_level

            # --- Frame Encoding and Emission ---
            try:
                _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                img_str = base64.b64encode(buffer).decode('utf-8')
                # Emit frame and the SMOOTHED lighting level
                socketio.emit('video_frame', {'image': img_str, 'lightingLevel': level_to_emit})
            except Exception as e:
                print(f"Error encoding or emitting frame: {e}")

            # --- FPS Control ---
            current_time = time.time()
            elapsed = current_time - last_frame_time
            wait_time = frame_delay - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
            last_frame_time = time.time() # Update last frame time accurately

        # --- Check for Stop/Switch/Pause signals ---
        if not play_event.is_set() and cap: # If stop signal received while playing
             print(f"Stop signal received for '{video_name}'. Pausing stream.")
             # --- Reset Smoothing State on Stop Request ---
             with lock:
                 stable_lighting_level = 0
                 pending_lighting_state = 0
                 frames_in_pending_state = 0
             # --- Reset TV State on Stop Request ---
             # Keep tv_detected_in_scene, but reset controlled state
             tv_current_state = "off" # Assume TV turns off when stopping
             socketio.emit('tv_status_update', {'status': 'unknown'}) # Reset frontend TV state
             # --- End Reset ---
             socketio.emit('video_status', {'status': 'stopped', 'video': video_name})

        # Short sleep if not playing to prevent busy-waiting when paused
        if not play_event.is_set():
            time.sleep(0.1)


    # --- Cleanup ---
    if cap:
        cap.release()
    video_processing_active = False
    # Reset TV state on thread exit
    socketio.emit('tv_status_update', {'status': 'unknown'})
    print("Video processing thread finished.")

# --- Frame Generation for Stream ---
# def generate(): ... # This function is no longer needed

# --- Flask Routes ---
@app.route("/")
def index():
    """Serves the main React application."""
    # Check if the build directory exists
    if not os.path.exists(app.template_folder) or not os.path.exists(os.path.join(app.template_folder, 'index.html')):
         return "Error: React build files not found in 'frontend/build'. Please run 'npm run build' in the 'frontend' directory.", 500
    return render_template("index.html")

@app.route("/api/videos")
def list_videos():
    """Returns a JSON list of available video names."""
    return jsonify(list(AVAILABLE_VIDEOS.keys()))

@app.route("/api/set_video", methods=['POST'])
def set_video():
    """API endpoint to switch the video source."""
    global next_video_path, switch_video_event, current_video_path, video_processing_active

    data = request.get_json()
    if not data or 'video_name' not in data:
        return jsonify({"error": "Missing 'video_name' in request"}), 400

    video_name = data['video_name']
    new_path = get_video_path(video_name)

    if not new_path:
        return jsonify({"error": f"Video '{video_name}' not found"}), 404

    with lock: # Check current state safely
        is_already_active = (new_path == current_video_path and not switch_video_event.is_set())
        is_processing = video_processing_active

    if is_already_active:
        return jsonify({"message": f"Video '{video_name}' is already the active source."}), 200

    if not is_processing:
        return jsonify({"error": "Video processing thread is not active"}), 503

    print(f"Request received to switch video to: {video_name}")
    next_video_path = new_path
    switch_video_event.set() # Signal the processing thread

    return jsonify({"message": f"Switching video to '{video_name}'"}), 200

# --- Static File Serving for React App ---
@app.route('/static/<path:path>')
def serve_static_files(path):
    return send_from_directory(app.static_folder, path)

@app.route('/manifest.json')
def serve_manifest():
    return send_from_directory(app.template_folder, 'manifest.json')

@app.route('/favicon.ico')
def serve_favicon():
     # Browsers often request this, serve it from the build root
     return send_from_directory(app.template_folder, 'favicon.ico', mimetype='image/vnd.microsoft.icon')

# --- Catch-all Route for React Router ---
# This must be the last route defined
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react_app(path):
    """Serves the React app for any non-API, non-static path."""
    # Check if path points to a static file first
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    # Check if path points to a file in the template folder root (like manifest.json, handled above but good fallback)
    elif os.path.exists(os.path.join(app.template_folder, path)):
         return send_from_directory(app.template_folder, path)
    else:
        # Otherwise, serve index.html for React Router to handle the route
        if not os.path.exists(os.path.join(app.template_folder, 'index.html')):
             return "Error: React index.html not found.", 500
        return send_from_directory(app.template_folder, 'index.html')

# --- SocketIO Event Handlers ---
@socketio.on('connect')
def handle_connect():
    """Handles new client connections."""
    print('Client connected:', request.sid)
    # Send initial state immediately - video is not playing yet
    with lock:
        video_path = current_video_path
        video_name = os.path.basename(video_path)
    # Emit 'ready' status if thread is alive, otherwise maybe 'stopped' or 'initializing'
    status = 'ready' if video_processing_active else 'stopped'
    socketio.emit('video_status', {'status': status, 'video': video_name}, room=request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    """Handles client disconnections."""
    print('Client disconnected:', request.sid)

@socketio.on('play_request')
def handle_play_request():
    """Handles request from client to start playing the video."""
    sid = request.sid
    print(f"Play request received from {sid}")
    if not video_processing_active:
         print("Warning: Play request received but video thread is not active.")
         socketio.emit('video_status', {'status': 'error', 'message': 'Processing backend not ready.'}, room=sid)
         return
    if not play_event.is_set():
        play_event.set() # Signal the thread to start/resume processing
        # Status update ('playing') will be sent by the thread itself
    else:
        print("Play request received but already playing.")

@socketio.on('stop_request')
def handle_stop_request():
    """Handles request from client to stop playing the video."""
    sid = request.sid
    print(f"Stop request received from {sid}")
    if play_event.is_set():
        play_event.clear() # Signal the thread to stop processing loop
        # Status update ('stopped') will be sent by the thread itself
    else:
        print("Stop request received but already stopped.")

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RoomAware Flask app with YOLOv8 video processing.")
    parser.add_argument(
        "--fps", type=float, default=None,
        help="Target FPS for processing. Uses video's original FPS if not set."
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Host address to bind the server to."
    )
    parser.add_argument(
        "--port", type=int, default=5000,
        help="Port number to run the server on."
    )
    parser.add_argument(
        "--debug", action='store_true',
        help="Run Flask in debug mode (use False for production/stability)."
    )
    args = parser.parse_args()

    # --- Initial Setup ---
    if not os.path.exists(VIDEO_FOLDER):
        os.makedirs(VIDEO_FOLDER)
        print(f"Created video folder: {VIDEO_FOLDER}")
        # You might want to add a check here if AVAILABLE_VIDEOS is empty

    # Verify default video exists and update if necessary
    with lock:
        default_path = current_video_path # Read initial default path
    if not os.path.exists(default_path):
        print(f"Warning: Default video '{DEFAULT_VIDEO_KEY}' not found at '{default_path}'.")
        found_alternative = False
        for name, path in AVAILABLE_VIDEOS.items():
            if os.path.exists(path):
                print(f"Using alternative video '{name}' ({path}) as default.")
                with lock: current_video_path = path # Update global default
                found_alternative = True
                break
        if not found_alternative:
            print("Error: No valid video files found in configuration or folder. Cannot start.")
            exit(1) # Exit if no videos are usable

    # --- Start Background Thread ---
    print("Starting video processing thread...")
    video_thread = threading.Thread(target=process_video, args=(args.fps,), daemon=True)
    video_thread.start()

    # --- Start Flask-SocketIO Server ---
    print(f"Starting Flask-SocketIO server on http://{args.host}:{args.port} ...")
    # Use socketio.run() instead of app.run()
    # debug=False is generally recommended with SocketIO/threading
    # allow_unsafe_werkzeug=True might be needed if using Werkzeug development server with threading/asyncio
    socketio.run(app, host=args.host, port=args.port, debug=False, use_reloader=False) # allow_unsafe_werkzeug=True

    # --- Cleanup on Exit (Ctrl+C) ---
    print("\nServer shutting down...")
    stop_event.set()
    if video_thread.is_alive():
        print("Waiting for video processing thread to finish...")
        video_thread.join(timeout=5.0)
        if video_thread.is_alive():
            print("Warning: Video processing thread did not exit cleanly.")
    print("Application finished.") 