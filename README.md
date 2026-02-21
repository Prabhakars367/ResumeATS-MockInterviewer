# 🤖 ResumeATS-MockInterviewer: AI-Powered Career Intelligence

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31019/)
[![TensorFlow 2.16](https://img.shields.io/badge/TensorFlow-2.16.1-orange.svg)](https://www.tensorflow.org/)
[![Keras 3](https://img.shields.io/badge/Keras-3.3.3-red.svg)](https://keras.io/)
[![Groq AI](https://img.shields.io/badge/LLM-Groq-green.svg)](https://groq.com/)

A state-of-the-art AI recruitment platform that combines **Real-time Behavioral Analytics**, **Generative AI Interviewing**, and **Job Market Intelligence**. This system doesn't just ask questions—it perceives candidate confidence, analyzes facial micro-expressions, and aligns technical evaluations with live job market data.

---

## 🏛️ Deep System Architecture

The system is built on a high-concurrency **FastAPI** backend, utilizing **Socket.IO** for low-latency full-duplex communication between the client's webcam and the ML analysis engine.

### 1. LLM Evaluation Engine (Powered by Groq)
We have migrated from traditional REST-based LLMs to **Groq's LPU™ Inference Engine**.
- **Model**: `llama-3.3-70b-versatile` (Primary), `llama-3-8b-8192` (Fallback).
- **Inference**: Sub-second latency for complex technical question generation and multi-dimensional answer evaluation.
- **Logic**: The system dynamically constructs prompts based on the candidate's Resume (parsed via customized PDF/DOCX logic) and the targeted Job Role.

### 2. Behavioral Perception System (CV)
Designed using the latest **Keras 3** and **MediaPipe** stacks for millisecond-level precision.
- **Emotion Recognition**: A Deep Convolutional Neural Network (CNN) trained on thousands of facial expressions, upgraded to run on **Keras 3** metadata for maximum performance. Detects **Happiness, Sadness, Anger, Neutrality, and Nervousness**.
- **Eye Tracking**: Utilizes **MediaPipe Face Mesh (468 landmarks)** to calculate Gaze Direction and Blink Rate, providing a "Focus Score" to detectable distractions.
- **Nervousness Index**: A proprietary algorithm combines facial symmetry, movement variance, and fear-indexed emotion probabilities to quantify candidate anxiety levels.

### 3. Automated Job Intelligence
Integrates a robust **Web Scraper** that monitors live job boards to ensure interview questions are relevant to current industry standards.
- Synchronizes with a local **SQLite** database (`jobs.db`) to provide candidates with real-time career opportunities post-interview.

---

## 🚀 Key Features

- **Dynamic Question Branching**: Adaptive technical and HR question flow.
- **Real-time Feedback Loop**: Instant analysis of speech-to-text transcripts vs. expected semantic benchmarks.
- **Automated Scorecards**: Comprehensive JSON-based reports combining behavioral metrics and technical accuracy.
- **Modern Web Interface**: Glassmorphic UI with real-time video overlay and metric visualizations.

---

## 🛠️ Technical Stack

| Component | Technology |
| :--- | :--- |
| **Backend** | FastAPI, Socket.IO, Python 3.10 |
| **LLM Gateway** | Groq SDK (LLama 3.1/3.3) |
| **Deep Learning** | Keras 3, TensorFlow 2.16.1 |
| **Computer Vision** | MediaPipe (Face Mesh), OpenCV |
| **Database** | SQLite, SQLAlchemy |
| **Natural Language** | TF-IDF, NLTK, Vosk/Google-SR |

---

## 📦 Environment Setup

### Prerequisites
- **Conda** (Recommended for ML dependency management)
- **Groq API Key** (Added to `.env`)

### Installation
1. **Clone and Enter**:
   ```bash
   git clone https://github.com/Prabhakars367/ResumeATS-MockInterviewer.git
   cd ResumeATS-MockInterviewer
   ```

2. **Environment Configuration**:
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_key_here
   ```

3. **Install Core ML Stack**:
   ```bash
   conda create -n ai_in python=3.10
   conda activate ai_in
   pip install -r requirements.txt
   ```

---

## 🏃 Running the Application

Start the production-ready server:
```bash
python app.py
```
Access the dashboard at `http://127.0.0.1:5000`.

---

## 🔧 Environment Stabilization Notes
The project is pinned to **TensorFlow 2.16.1** and **Keras 3.3.3**. This specific configuration is required to support both the custom **Emotion Recognition Model** and **MediaPipe** landmarks simultaneously. **Protobuf 4.25.3** is enforced to prevent version conflicts common in modern ML pipelines.

---

## 📄 License
[MIT License](LICENSE) - See LICENSE file for details.
