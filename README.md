
# AI Resume & Interview Assistant

This project is a powerful resume analysis and mock interview application built using **Streamlit** and **Natural Language Processing (NLP)** techniques. It extracts key details from resumes, matches them to relevant job roles, and simulates both technical and HR interviews using AI-generated questions and feedback.

---

## 🚀 Features

- 📄 Parses resumes from PDF files  
- 🔍 Matches resumes to job roles using semantic similarity (FAISS + Sentence Transformers)  
- ✍️ Supports optional custom job descriptions  
- 🧠 Simulates technical and HR interviews using **Google Gemini (Generative AI)**  
- 📝 Generates personalized feedback on candidate answers  

---

## 📦 Dataset

- Uses the **Structured Resume Dataset** from Kaggle for job-role matching:  
  [Structured Resume Dataset on Kaggle](https://www.kaggle.com/datasets/suriyaganesh/resume-dataset-structured)

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit  
- **Resume Parsing**: PyMuPDF  
- **Embeddings**: Sentence Transformers  
- **Similarity Search**: FAISS  
- **AI Questioning & Feedback**: Google Gemini  
- **Data Handling**: Pandas, Scikit-learn  

---

## ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Prabhakars367/ResumeATS-MockInterviewer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your Gemini API key**  
   Open `llm.py` and configure your API key:
   ```python
   configure(api_key="your-gemini-api-key")
   ```

4. **Download and preprocess the dataset**
   - Download from: https://www.kaggle.com/datasets/suriyaganesh/resume-dataset-structured
   - Place the raw files in the project directory.
   - Run:
     ```bash
     python preprocess.py
     ```

5. **Launch the application**
   ```bash
   streamlit run app.py
   ```

---

## 📁 File Structure

```
├── app.py              # Main Streamlit application
├── llm.py              # Gemini API interaction (Q&A + Feedback)
├── model1.py           # Resume-job matching with FAISS
├── preprocess.py       # Preprocessing Kaggle dataset
├── final_cleaned.csv   # Final dataset used for job-role matching
├── requirements.txt    # Python dependencies
```

---

## 💡 How It Works

1. **Upload Resume**: A user uploads their resume in PDF format.  
2. **Resume Parsing**: The app extracts sections like education, experience, and skills.  
3. **Job Matching**: It matches the resume to the most suitable role or a custom job description using embeddings + FAISS.  
4. **Interview Simulation**: Generates and presents both technical and HR interview questions.  
5. **Feedback Loop**: User answers are evaluated by Gemini, which provides AI-generated feedback.

---

## 📬 Contact

For questions or collaboration opportunities, feel free to connect:

- **GitHub**: [@Prabhakars367](https://github.com/Prabhakars367)  
- **LinkedIn**: [Prabhakar Kumar Singh](https://www.linkedin.com/in/prabhakars367/)  
- **Email**: prabhakars367@gmail.com

---

## 🧠 Contributions

Contributions, issues, and feature requests are welcome!  
Feel free to check the [issues page](https://github.com/your-username/ai-resume-interview-assistant/issues) if you want to contribute.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
