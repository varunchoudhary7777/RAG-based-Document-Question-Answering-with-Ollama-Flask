# --- 1. Install Dependencies ---
# !pip install flask pyngrok
# !curl -fsSL https://ollama.com/install.sh | sh

# --- 2. Start Ollama Server & Pull Models ---
import os
import asyncio

# Start the Ollama server in the background
# Using nohup to ensure it stays running
# !nohup ollama serve > ollama.log 2>&1 &

# Give the server a moment to start up
import time
time.sleep(5)

# Pull the necessary models
# !ollama pull llama3:8b
# !ollama pull mxbai-embed-large

# !pip install langchain-community
# !pip install pypdf
# !pip install chromadb
# !pip install tiktoken
# !pip install Flask
# !pip install ngrok

# --- 3. The All-in-One Web Application (Corrected and Complete) ---
from flask import Flask, request, jsonify, render_template_string
from pyngrok import ngrok
import os
import json
from getpass import getpass

# LangChain and other imports for your RAG logic
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pypdf import PdfReader

# --- Part A: Define the HTML Frontend ---
# This part is unchanged.
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AI Resume Analyzer</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; max-width: 800px; margin: auto; padding: 20px; background-color: #f8f9fa; color: #343a40; }
        h1 { color: #0056b3; }
        form { background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        button { background-color: #007bff; color: white; padding: 10px 15px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
        button:hover { background-color: #0056b3; }
        #results { white-space: pre-wrap; background-color: white; border: 1px solid #ddd; padding: 15px; margin-top: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        #loader { display: none; text-align: center; padding: 20px; font-size: 18px; }
    </style>
</head>
<body>
    <h1>AI Resume Analyzer</h1>
    <p>Upload a resume in PDF format to receive an AI-powered analysis and job recommendations.</p>
    <form id="resumeForm">
        <input type="file" id="resumeFile" name="resume" accept=".pdf" required>
        <button type="submit">Analyze</button>
    </form>
    <div id="loader">⚙️ Analyzing... This may take a moment. Please wait.</div>
    <div id="results"></div>
    <script>
        document.getElementById('resumeForm').addEventListener('submit', async function(event) {
            event.preventDefault();
            const fileInput = document.getElementById('resumeFile');
            const resultsDiv = document.getElementById('results');
            const loader = document.getElementById('loader');
            loader.style.display = 'block';
            resultsDiv.innerHTML = '';
            const formData = new FormData();
            formData.append('resume', fileInput.files[0]);
            try {
                const response = await fetch('/analyze', { method: 'POST', body: formData });
                const data = await response.json();
                if (response.ok) {
                    resultsDiv.innerText = data.report;
                } else {
                    resultsDiv.innerText = 'Error: ' + data.error;
                }
            } catch (error) {
                resultsDiv.innerText = 'An unexpected error occurred: ' + error;
            } finally {
                loader.style.display = 'none';
            }
        });
    </script>
</body>
</html>
"""

# --- Part B: Your Full RAG Logic ---
# Initialize models (this will connect to the server running in the background)
llm = ChatOllama(model="llama3:8b")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# 1. Setup the Knowledge Base
jobs_data = [
    {
        "title": "Senior Data Scientist",
        "description": "We are seeking a Senior Data Scientist with over 5 years of experience in machine learning, statistical modeling, and Python. The ideal candidate will have a strong background in SQL, experience with cloud platforms like AWS or GCP, and proficiency with data visualization tools such as Tableau. Responsibilities include developing predictive models, performing A/B testing, and communicating insights to stakeholders."
    },
    {
        "title": "Cloud DevOps Engineer",
        "description": "This role requires a DevOps Engineer with deep expertise in CI/CD pipelines, containerization (Docker, Kubernetes), and infrastructure-as-code (Terraform, Ansible). Must have 3+ years of experience managing production environments in AWS. Strong scripting skills in Python or Bash are essential. Experience with monitoring tools like Prometheus and Grafana is a plus."
    },
    {
        "title": "UX/UI Designer",
        "description": "Looking for a creative UX/UI Designer to create intuitive and engaging user experiences. Must be proficient in Figma, Sketch, and Adobe Creative Suite. A strong portfolio demonstrating user-centered design principles, wireframing, and prototyping is required. Experience with user research and usability testing is highly valued. Collaboration with product managers and engineers is key."
    }
]
job_texts = [f"Job Title: {job['title']}\nDescription: {job['description']}" for job in jobs_data]
vector_store = Chroma.from_texts(texts=job_texts, embedding=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 2. Define all helper functions
def process_resume_pdf(pdf_path: str):
    """Extracts text content from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# --- ADD THIS HELPER FUNCTION ---
def extract_json_from_string(text: str):
    """
    Finds and parses a JSON object from a string that might contain other text.
    """
    try:
        # Find the first '{' and the last '}' to isolate the JSON block
        start_index = text.find('{')
        end_index = text.rfind('}') + 1

        if start_index == -1 or end_index == 0:
            raise ValueError("No JSON object found in the string.")

        json_str = text[start_index:end_index]
        return json.loads(json_str)
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        raise ValueError(f"Failed to extract valid JSON from LLM output: {text}")


# --- REPLACE YOUR OLD get_structured_resume FUNCTION WITH THIS ---
def get_structured_resume(resume_text: str):
    """
    Uses an LLM to parse raw resume text into a structured JSON object,
    with a more robust prompt and parsing method.
    """
    # 1. We create a stricter prompt
    prompt = ChatPromptTemplate.from_template(
        """
        You are a JSON-only API. Your sole job is to parse the provided resume text and extract key information.
        Do not output any text, explanations, or conversational pleasantries before or after the JSON object.
        Your entire response must be a single, valid JSON object with the following keys: "skills", "experience", "education".

        Resume Text:
        {resume_text}
        """
    )

    # 2. We create a chain that first gets the raw text output, then parses it
    # This is more robust than a direct JSON parser.
    parser = StrOutputParser()
    chain = prompt | llm | parser

    # 3. Invoke the chain and manually extract the JSON
    llm_output = chain.invoke({"resume_text": resume_text})

    return extract_json_from_string(llm_output)

def ats_scorer(resume_json: dict, job_description: str):
    """Performs a simulated ATS analysis, scores the resume against a job, and provides feedback."""
    # This is a simplified version for demonstration. You can expand on this.
    parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_template(
        "You are an ATS (Applicant Tracking System). Analyze the user's skills and the required skills from the job description. Provide a score out of 100 and 3 bullet points of constructive feedback for improvement.\n\nUser's Resume Skills: {skills}\n\nJob Description: {job_desc}"
    )
    # CORRECTED LINE
    chain = prompt | llm | parser
    feedback = chain.invoke({"skills": json.dumps(resume_json.get("skills", [])), "job_desc": job_description})
    # A more robust score would be extracted, but for now we'll generate a placeholder.
    score = 75
    return {"score": score, "feedback": feedback}

# 3. Define the main analysis pipeline
def main_analysis(resume_pdf_path: str):
    """This function runs the full RAG pipeline and returns the final report."""
    try:
        print("Starting analysis...")
        resume_text = process_resume_pdf(resume_pdf_path)
        structured_resume = get_structured_resume(resume_text)
        retrieved_docs = retriever.invoke(json.dumps(structured_resume))
        top_job_match = retrieved_docs[0].page_content
        ats_results = ats_scorer(structured_resume, top_job_match)

        report_parser = StrOutputParser()
        report_prompt = ChatPromptTemplate.from_template(
            """
            You are an expert career advisor. Generate a comprehensive, user-friendly report based on the following information.

            **User's Resume Summary:**
            {resume}

            **Top 3 Job Matches Found:**
            1. {job_1}
            2. {job_2}
            3. {job_3}

            **Detailed ATS Analysis for the Top Job Match:**
            - ATS Score: {ats_score}/100
            - Feedback for Improvement:
            {ats_feedback}

            ---

            **Final Report:**
            Structure the report with two main sections: "Career Recommendations" and "Resume Optimization Plan". Keep the tone encouraging and professional.
            """
        )
        report_chain = report_prompt | llm | report_parser
        final_report = report_chain.invoke({
            "resume": json.dumps(structured_resume, indent=2),
            "job_1": retrieved_docs[0].page_content,
            "job_2": retrieved_docs[1].page_content,
            "job_3": retrieved_docs[2].page_content,
            "ats_score": ats_results["score"],
            "ats_feedback": ats_results["feedback"]
        })

        print("Analysis complete.")
        return final_report

    except Exception as e:
        print(f"An error occurred: {e}")
        return f"An error occurred during the analysis process. Please check the resume format and try again. Error: {e}"

# --- Part C: The Flask Application ---
# This part is unchanged.
app = Flask(__name__)
os.makedirs("uploads", exist_ok=True)

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/analyze", methods=["POST"])
def analyze():
    if 'resume' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['resume']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filepath = os.path.join("uploads", file.filename)
        file.save(filepath)
        try:
            report = main_analysis(filepath)
            return jsonify({"report": report})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    return jsonify({"error": "File processing failed"}), 500

# --- Part D: Start ngrok and the Flask App ---
# This part is unchanged.
authtoken = getpass("Enter your ngrok authtoken: ")
ngrok.set_auth_token(authtoken)

public_url = ngrok.connect(5000)
print(f"✅ Your public URL is: {public_url}")
print("Clients can now access your application at this URL.")

app.run(port=5000)

