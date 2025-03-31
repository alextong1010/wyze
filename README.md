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

# To Run:
```bash
npm run build
python main.py
```
