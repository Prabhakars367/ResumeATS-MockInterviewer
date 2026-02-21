# AI Smart Interviewer

A comprehensive AI-powered interview application designed to conduct technical interviews, analyze candidate responses, and monitor behavioral cues using computer vision.

## 🚀 Key Features

*   **Intelligent Questioning**: Selects technical questions based on job roles (e.g., Python Developer, Data Scientist) or randomizes them.
*   **Voice Interaction**: Uses Speech Recognition to listen to candidate answers and Text-to-Speech (TTS) to ask questions and provide feedback.
*   **Response Analysis**:
    *   **Content Matching**: Uses TF-IDF and Cosine Similarity to compare user answers with expected answers.
    *   **Scoring**: Provides an immediate similarity score (0.0 - 1.0) and feedback (Excellent, Good, Fair, Needs Improvement).
*   **Behavioral Monitoring**:
    *   **Emotion Recognition**: Real-time facial emotion detection (Happy, Sad, Neutral, Fear, etc.) using a Deep Learning model (`emotion_recognition_model.h5`).
    *   **Eye Tracking**: Tracks gaze direction (Center, Up, Down, Left, Right) to detect distractions or "looking away" events using MediaPipe Face Mesh.
*   **Resume Screening**: Built-in resume parser (PDF/DOCX) to extract text and potentially evaluate candidate fit.
*   **Web Interface**: Full-featured web application built with Flask and Socket.IO for real-time video processing.
*   **Detailed Reporting**: Saves interview logs, emotion summaries, and eye-tracking data to JSON files for post-interview analysis.

## 🛠️ Technology Stack

*   **Backend**: Python, Flask, Flask-SocketIO
*   **Computer Vision**: OpenCV, MediaPipe, TensorFlow/Keras
*   **NLP & Analytics**: NLTK, Scikit-learn, Pandas, NumPy
*   **Speech**: SpeechRecognition (Google/Vosk), PyDub
*   **Frontend**: HTML, JavaScript (Socket.IO client)

## 📋 Prerequisites

*   **Python 3.8+**
*   **Visual Studio C++ Build Tools**: Required for compiling some Python packages like `dlib` or `pyaudio` (if used).
*   **Webcam**: Required for emotion and eye tracking.
*   **Microphone**: Required for voice interaction.

## 📦 Installation

1.  **Clone the repository** (if applicable):
    ```bash
    git clone <repository-url>
    cd emotion_rec
    ```

2.  **Create a Virtual Environment** (Recommended):
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

    *Note: If you encounter issues with `pyaudio`, you may need to install the pre-compiled wheel files for your specific Python version.*

4.  **Download NLTK Data**:
    The application will attempt to download necessary NLTK data (`punkt`, `stopwords`) automatically on first run.

## 🏃 Usage

### Option 1: Web Application (Recommended)
This runs the full interactive web interface.

1.  Run the Flask application:
    ```bash
    python app.py
    ```
2.  Open your browser and navigate to:
    `http://127.0.0.1:5000` or `http://localhost:5000`
3.  Grant permissions for Camera and Microphone when prompted.

### Option 2: Standalone Script
For a CLI-based interview experience with a CV2 window:

1.  Run the main script:
    ```bash
    python main.py
    ```
2.  Follow the voice/text prompts to select a job role.
3.  Speak your answers clearly when prompted.
4.  Press 'q' to quit the camera window if needed.

## 📂 Project Structure

*   `app.py`: Main Flask application entry point. Handles routes, Socket.IO events, and the web-based interview logic.
*   `main.py`: Standalone Python script for running the interview process locally without a web browser.
*   `eye_tracking.py`: Module for gaze estimation and eye tracking using MediaPipe.
*   `face_emotion_system.py`: Module handling the emotion recognition model and video stream processing.
*   `voice_rec.py`: Handles Speech-to-Text and Text-to-Speech functionality.
*   `models/`: Directory for storing models (e.g., Vosk language model).
*   `templates/`: HTML templates for the Flask app (`index.html`, `interview.html`).
*   `final_cleaned.csv` / `final_csv.csv`: Dataset containing technical interview questions and answers.
*   `requirements.txt`: List of Python dependencies.

## 🔧 Troubleshooting

*   **ffmpeg not found**: Ensure `ffmpeg` is installed and added to your system PATH if using local audio processing features.
*   **Vosk Model**: If using offline speech recognition, ensure the Vosk model is extracted to `models/vosk-small-en-us-0.15`.
*   **Camera Error**: Ensure no other application is using the webcam.

## 📄 License

[Include License Information Here]
