# This file has been renamed from main.py to app.py

import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi import Form
from pydantic import BaseModel
import fitz  # PyMuPDF
import docx
from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI
import whisper
import coqui_tts
from smolagents import SmolAgent
from dotenv import load_dotenv  # Import dotenv
import gradio as gr

load_dotenv()  # Load environment variables from .env file

app = FastAPI()

class UserInput(BaseModel):
    user_input: str

class JobContext(BaseModel):
    job_role: str
    job_description: str

# Initialize the OpenAI LLM with your API key from the environment variable
llm = OpenAI(api_key=os.getenv('API_KEY'))  # Use the API key from .env

# Initialize Whisper model for STT
whisper_model = whisper.load_model('base')

# Initialize Coqui TTS model
tts_model = coqui_tts.TTS(model_name='coqui-tts')

# Create a prompt template for adaptive questioning
prompt_template = PromptTemplate(
    input_variables=["user_input", "job_role", "job_description"],
    template="Based on the user's previous responses: {user_input}, job role: {job_role}, job description: {job_description}, generate an adaptive interview question."
)

# Create a LangChain LLM chain
chain = LLMChain(llm=llm, prompt=prompt_template)

# Initialize SmolAgent
smol_agent = SmolAgent()

@app.get("/status")
async def read_status():
    return {"status": "ok"}

@app.post("/generate-interview")
async def generate_interview(user_input: UserInput, job_context: JobContext):
    recent_context = user_input.user_input

    # Use SmolAgent to determine the next question based on the current context
    next_question = smol_agent.decide_next_question(recent_context)

    # Generate the adaptive prompt using LangChain
    adaptive_prompt = chain.run(user_input=recent_context, job_role=job_context.job_role, job_description=job_context.job_description)

    # Combine the SmolAgent's decision with the adaptive prompt
    generated_text = f"{next_question} {adaptive_prompt}"
    return {"generated_text": generated_text}

def transcribe_audio(audio_path: str) -> str:
    # Use Whisper to transcribe audio to text
    result = whisper_model.transcribe(audio_path)
    return result['text']

def synthesize_speech(text: str, output_path: str):
    # Use Coqui TTS to convert text to speech
    tts_model.save_to_file(text, output_path)

@app.post("/process-audio")
async def process_audio(audio: UploadFile = File(...)):
    # Save the uploaded audio file temporarily
    audio_path = f"/tmp/{audio.filename}"
    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    # Transcribe the audio to text
    transcribed_text = transcribe_audio(audio_path)

    # Process the text response through the existing Gemini API/LangChain adaptive logic
    # Assuming job_context is available or passed in some way
    job_context = JobContext(job_role="example_role", job_description="example_description")
    adaptive_prompt = chain.run(user_input=transcribed_text, job_role=job_context.job_role, job_description=job_context.job_description)

    # Synthesize speech from the generated feedback
    output_audio_path = f"/tmp/feedback_audio.wav"
    synthesize_speech(adaptive_prompt, output_audio_path)

    return {"audio_url": output_audio_path}

@app.post("/submit-context")
async def submit_context(job_role: str = Form(...), job_description: str = Form(...), resume: UploadFile = File(...)):
    if resume.content_type not in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF and DOCX are allowed.")

    resume_text = ""
    if resume.content_type == "application/pdf":
        pdf_document = fitz.open(stream=await resume.read(), filetype="pdf")
        resume_text = " ".join([page.get_text() for page in pdf_document])
    elif resume.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(resume.file)
        resume_text = " ".join([para.text for para in doc.paragraphs])

    # Here you would implement parsing logic to extract skills, experience, etc.
    # For now, we will just return the raw text for demonstration.
    return {"job_role": job_role, "job_description": job_description, "resume_text": resume_text}

# Gradio function for status

def gradio_read_status():
    return read_status()

# Gradio function for processing audio

def gradio_process_audio(audio):
    audio_path = f'/tmp/{audio.name}'
    with open(audio_path, 'wb') as buffer:
        buffer.write(audio.read())
    return process_audio(audio=audio)

# Gradio function for submitting context

def gradio_submit_context(job_role, job_description, resume):
    return submit_context(job_role=job_role, job_description=job_description, resume=resume)

# Set up Gradio interfaces

iface_audio = gr.Interface(fn=gradio_process_audio, inputs=gr.inputs.Audio(), outputs='audio', title='Audio Processing')
iface_context = gr.Interface(fn=gradio_submit_context, inputs=[gr.inputs.Textbox(label='Job Role'), gr.inputs.Textbox(label='Job Description'), gr.inputs.File(label='Resume')], outputs='text', title='Submit Context')

# Launch Gradio app
if __name__ == '__main__':
    iface_audio.launch()
    iface_context.launch() 