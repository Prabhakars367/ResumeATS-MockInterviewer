import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Google API Key is missing! Please set it in the environment variables.")
genai.configure(api_key=GOOGLE_API_KEY)

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Function to create a FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Load the conversational QA chain
def get_conversational_chain():
    prompt_template = """
    Given the extracted resume text and job description, evaluate the ATS (Applicant Tracking System) score based on industry standards.
    Also, generate relevant interview questions related to the job role and HR questions based on the resume.
    
    Resume Text:
    {context}
    
    Job Description:
    {job_description}
    
    Generate:
    - ATS Score (Out of 100)
    - Relevant technical and HR questions for the interview.
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "job_description"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Function to process resume and get ATS score and questions
def analyze_resume(resume_text, job_description):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(resume_text)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "job_description": job_description}, return_only_outputs=True)
        st.write("### ATS Score:", response["output_text"].split("\n")[0])
        st.write("### Suggested Interview Questions:")
        st.write("\n".join(response["output_text"].split("\n")[1:]))
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Main Streamlit app
def main():
    st.set_page_config(page_title="Resume ATS & Interview Prep", layout="wide")
    st.title("📄 Resume ATS Score & Interview Questions Generator")
    
    job_description = st.text_area("Enter the Job Description for Resume Evaluation:")
    pdf_docs = st.file_uploader("Upload your Resume (PDF Format)", accept_multiple_files=False, type='pdf')
    
    if st.button("Analyze Resume"):
        if pdf_docs and job_description:
            with st.spinner("Processing your resume..."):
                raw_text = get_pdf_text([pdf_docs])
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                analyze_resume(raw_text, job_description)
                st.success("✅ Resume analysis completed!")
        else:
            st.warning("⚠️ Please upload a resume and enter a job description.")
    
    if st.button("Clear Data"):
        st.experimental_rerun()

if __name__ == "__main__":
    main()
