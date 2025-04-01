## AI Model and Algorithm Explanation

Our project leverages several advanced AI models and algorithms to provide a seamless and intelligent home automation experience. Below is an overview of the key components:

1. **Object Detection with OpenCV:**
   We use OpenCV for object detection in our video feed. This allows us to locate and identify various objects in the room, such as people, the TV, and other items. OpenCV's robust computer vision capabilities enable real-time detection and tracking, which is crucial for responsive home automation.

2. **Facial Recognition with face_recognition:**
   The `face_recognition` Python package is employed for facial detection and recognition. This enables us to detect custom users and build personalized user profiles. By recognizing individual users, we can adjust room settings based on their preferences, such as lighting levels and TV state. This personalized approach enhances the user experience and provides a tailored environment.

3. **Text-to-Speech and Agentic Orchestration with ElevenLabs and Gemini-based System:**
   We integrate ElevenLabs' Text-to-Speech technology to provide natural and clear voice interactions, allowing the system to communicate with users effectively by providing updates, notifications, and responses to user commands. This is seamlessly managed and coordinated by our Gemini-based agentic orchestration system, which ensures that all AI components, including the Text-to-Speech functionality, work together harmoniously. This orchestration provides a cohesive and efficient home automation experience, enhancing user interaction and system responsiveness.

By combining these technologies, our project offers a sophisticated and user-friendly solution for intelligent home automation, making everyday tasks more convenient and personalized.


## Setup and Installation

Follow these steps to set up and run the project on macOS or Linux.

### Prerequisites

1.  **Python 3:** Ensure you have Python 3 installed. You can check with `python3 --version`.
2.  **Node.js and npm:** Install Node.js (which includes npm) and build the frontend.
    *   **Install Node.js & npm:**
        *   **macOS (using Homebrew):**
            ```bash
            brew install node
            ```
        *   **Ubuntu/Debian:**
            ```bash
            sudo apt update
            sudo apt install nodejs npm
            ```
        *   Verify installation:
            ```bash
            node -v
            npm -v
            ```
    *   **Build Frontend:** Navigate to the `frontend` directory and run the build:
        ```bash
        cd frontend
        npm install  # Install dependencies
        # npm start  # Optional: Run the frontend development server
        npm run build # Build the production frontend
        ```

### Backend Setup (Python)

1.  **Navigate to the project root directory.**
2.  **(Recommended) Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # On Windows use: venv\Scripts\activate
    ```
3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The first time you run the application, the YOLOv8 model (`yolov8n.pt`) will be downloaded automatically if it's not present in the root directory.*

pip install opencv-python
pip install numpy
pip install flask
pip install ultralytics
pip install face_recognition
pip install google.generativeai

# To Run:
```bash
cd frontend
npm run build
```
Open another terminal session and run:
```bash
python main.py
```
