"""
llm.py — Groq SDK integration for the AI Interviewer system.
Replaces Gemini with Groq for faster and more reliable inference.
"""

import os
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# 🔹 Groq API key setup
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("[llm.py] WARNING: GROQ_API_KEY not set in environment. Groq calls will fail.")

# Model configuration
MODEL_PRIMARY = "llama-3.3-70b-versatile"
MODEL_FALLBACK = "llama-3-8b-8192"

client = None
if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)

# === Helper Function ===
def _groq_call(prompt: str, model: str = MODEL_PRIMARY, retries: int = 3, delay: int = 2):
    """Send a prompt to Groq API using the SDK."""
    if not client:
        raise RuntimeError("Groq client not configured")

    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1024,
            )
            
            text = completion.choices[0].message.content
            if text:
                return text.strip()
            
            raise ValueError("Empty response from Groq")

        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                print(f"[Groq] Rate limit hit, retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
                continue
            
            print(f"[Groq Error] {e}")
            if attempt == retries - 1:
                if model != MODEL_FALLBACK:
                    print("[Groq] Switching to fallback model...")
                    return _groq_call(prompt, model=MODEL_FALLBACK, retries=1)
                raise e
            time.sleep(delay)

    raise RuntimeError("Groq API failed after retries.")

# === Technical Questions ===
def get_technical_questions(role: str):
    prompt = (
        f"Generate 5 concise, real-world technical interview questions for a {role}. "
        "Each question should test applied understanding. Return only questions, separated by new lines. "
        "Do not include numbers or any introductory text."
    )
    try:
        print(f"[Groq] Generating technical questions for '{role}'...")
        text = _groq_call(prompt)
        return [line.strip() for line in text.split("\n") if line.strip()]
    except Exception as e:
        print(f"[Groq Exception - technical] {e}")
        return [
            f"What are the main skills required for a {role}?",
            f"Describe a challenging project you handled as a {role}.",
            "How do you troubleshoot performance issues?",
            "What are the best practices you follow in your work?",
            "Explain a recent technology advancement in your field."
        ]

# === HR Questions ===
def get_hr_questions():
    prompt = (
        "Generate 5 HR interview questions that assess communication, teamwork, "
        "leadership, and motivation. Return only questions, separated by new lines. "
        "Do not include numbers or any introductory text."
    )
    try:
        print("[Groq] Generating HR questions...")
        text = _groq_call(prompt)
        return [line.strip() for line in text.split("\n") if line.strip()]
    except Exception as e:
        print(f"[Groq Exception - HR] {e}")
        return [
            "Tell me about yourself.",
            "Why do you want to work with our company?",
            "Describe a time you resolved a team conflict.",
            "What motivates you to perform your best at work?",
            "Where do you see yourself in 5 years?"
        ]

# === Candidate Answer Evaluation ===
def evaluate_candidate_answers(questions, answers, round_type="technical"):
    """Ask Groq to evaluate answers and give feedback."""
    joined_qas = "\n".join([f"Q: {q}\nA: {a}" for q, a in zip(questions, answers)])
    prompt = (
        f"You are an expert interviewer. Evaluate these {round_type} answers.\n"
        "For each question, rate the answer (1–10) and provide one brief feedback line.\n\n"
        f"{joined_qas}"
    )
    try:
        print("[Groq] Evaluating candidate answers...")
        return _groq_call(prompt)
    except Exception as e:
        print(f"[Groq Exception - evaluation] {e}")
        return "Feedback not available due to API error."
