# Simplified app.py for Hugging Face Spaces
# Focusing purely on Gradio interface

import os
import shutil
import tempfile
import uuid
from dotenv import load_dotenv
import numpy as np
import scipy.io.wavfile as wav
import gradio as gr
import fitz  # PyMuPDF
import docx
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# API key handling - prioritize environment variables
# First check for API_KEY directly in environment (for Hugging Face Spaces)
api_key = os.environ.get('API_KEY')

# If not found, try to load from .env file
if not api_key:
    # Look in current directory and parent directory for .env file
    for env_path in ['.', '..', '../..']:
        env_file = os.path.join(env_path, '.env')
        if os.path.exists(env_file):
            load_dotenv(env_file)
            api_key = os.environ.get('API_KEY')
            print(f"Loaded API key from {env_file}")
            break

# If still not found, check for .env.example as a last resort
if not api_key:
    for env_path in ['.', '..', '../..']:
        env_example = os.path.join(env_path, '.env.example')
        if os.path.exists(env_example):
            load_dotenv(env_example)
            api_key = os.environ.get('API_KEY')
            print(f"WARNING: Using example API key from {env_example}. This is not secure for production!")
            break

# Final check - if no API key found, provide clear error
if not api_key:
    print("ERROR: No API key found. Please set the API_KEY environment variable.")
    print("For Hugging Face Spaces, set this in the Settings > Repository Secrets section.")
    # We'll still initialize with a placeholder, but it won't work for actual requests
    api_key = "MISSING_API_KEY"

# Create temp directory if it doesn't exist
TEMP_DIR = os.path.join(os.getcwd(), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# Initialize the Google Gemini LLM with the API key
llm = ChatGoogleGenerativeAI(model="gemini-pro", api_key=api_key)

# Create a prompt template for adaptive questioning
prompt_template = PromptTemplate(
    input_variables=["user_input", "job_role", "job_description"],
    template="Based on the user's previous responses: {user_input}, job role: {job_role}, job description: {job_description}, generate an adaptive interview question."
)

# Create a chain using the newer langchain-core approach
chain = prompt_template | llm | StrOutputParser()

# Simple agent to decide next questions
class SimpleAgent:
    def decide_next_question(self, context):
        return "Tell me more about your experience with this role."

# Initialize SimpleAgent
smol_agent = SimpleAgent()

# Helper functions
def transcribe_audio(audio_path: str) -> str:
    # Simplified transcription function that returns a placeholder
    return "This is a placeholder transcription. In a real implementation, this would be the transcribed text from the audio file."

def synthesize_speech(text: str, output_path: str):
    # Generate a simple sine wave as a placeholder
    sample_rate = 44100  # 44.1kHz
    duration = 3  # seconds
    frequency = 440  # A4 note
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
    wav.write(output_path, sample_rate, audio_data.astype(np.float32))

# Gradio handler functions
def gradio_process_audio(audio_data):
    if audio_data is None:
        return None
    
    audio_path = None
    try:
        # Create temporary files
        audio_path = os.path.join(TEMP_DIR, f"input_{uuid.uuid4()}.wav")
        output_path = os.path.join(TEMP_DIR, f"output_{uuid.uuid4()}.wav")
        
        # Save the input audio
        sample_rate, audio_array = audio_data
        wav.write(audio_path, sample_rate, audio_array)
        
        # Transcribe the audio
        transcription = transcribe_audio(audio_path)
        
        # Generate a response (in a real implementation, this would use an LLM)
        response_text = f"I heard: {transcription}. How can I help you with your interview preparation?"
        
        # Synthesize speech for the response
        synthesize_speech(response_text, output_path)
        
        # Return the audio file path for Gradio to play
        return output_path
    
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None
    
    finally:
        # Clean up temporary files
        if audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except:
                pass

def gradio_submit_context(job_role, job_description, resume_file):
    if not resume_file:
        return "Please upload a resume file."
    
    try:
        resume_text = ""
        file_path = resume_file.name
        
        if file_path.endswith('.pdf'):
            # Use PyMuPDF for PDF processing
            pdf_document = fitz.open(file_path)
            resume_text = " ".join([page.get_text() for page in pdf_document])
        elif file_path.endswith('.docx'):
            doc = docx.Document(file_path)
            resume_text = " ".join([para.text for para in doc.paragraphs])
        else:
            return "Invalid file type. Only PDF and DOCX are allowed."
        
        return f"Job Role: {job_role}\nJob Description: {job_description}\nResume Text: {resume_text[:500]}..."
    except Exception as e:
        return f"Error processing resume: {e}"

def gradio_generate_interview(user_input, job_role, job_description):
    if api_key == "MISSING_API_KEY":
        return "ERROR: No valid API key found. Please set the API_KEY in Hugging Face Spaces settings."
        
    try:
        # Generate the adaptive prompt using the chain
        adaptive_prompt = chain.invoke({
            "user_input": user_input, 
            "job_role": job_role, 
            "job_description": job_description
        })
        return adaptive_prompt
    except Exception as e:
        return f"Error generating interview question: {e}"

# Create the Gradio interface
def create_interface():
    with gr.Blocks(title="AI Interviewer") as interface:
        gr.Markdown("""
        # AI Interview Coaching Agent
        
        This application helps with interview preparation by providing audio processing, 
        resume analysis, and adaptive interview questions.
        """)
        
        with gr.Tab("Audio Processing"):
            audio_input = gr.Audio(label="Record your voice")
            audio_output = gr.Audio(label="AI Response")
            audio_button = gr.Button("Process Audio")
            audio_button.click(fn=gradio_process_audio, inputs=audio_input, outputs=audio_output)
            
        with gr.Tab("Submit Context"):
            job_role = gr.Textbox(label="Job Role")
            job_desc = gr.Textbox(label="Job Description")
            resume = gr.File(label="Resume")
            submit_button = gr.Button("Submit")
            output = gr.Textbox(label="Result")
            submit_button.click(fn=gradio_submit_context, inputs=[job_role, job_desc, resume], outputs=output)
        
        with gr.Tab("Generate Interview Question"):
            user_input = gr.Textbox(label="Your Previous Responses")
            job_role_input = gr.Textbox(label="Job Role")
            job_desc_input = gr.Textbox(label="Job Description")
            generate_button = gr.Button("Generate Question")
            question_output = gr.Textbox(label="Interview Question")
            generate_button.click(
                fn=gradio_generate_interview, 
                inputs=[user_input, job_role_input, job_desc_input], 
                outputs=question_output
            )
    
    return interface

# Create the demo interface for Hugging Face Spaces
demo = create_interface()

# Launch the app when run directly (for testing locally)
if __name__ == "__main__":
    demo.launch()
