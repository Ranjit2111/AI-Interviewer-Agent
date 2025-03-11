---

   title: "AI Interview Coaching Agent"

   emoji: "🚀"  

   colorFrom: "indigo"  

   colorTo: "yellow"  

   sdk: "gradio" 

   sdk_version: "3.32.0"  

   app_file: "app.py"  

   pinned: false

---

# Gradio API for Interview Assistant

This is a Gradio-based API for an interview assistant that processes audio inputs and submits job context information.

## Dependency Changes

The project has been modified to resolve dependency conflicts:

1. Replaced the full `langchain` package with specific components:
   - `langchain-core>=0.1.28`
   - `langchain-community>=0.0.10`
   - `langchain-google-genai>=0.0.3`

2. Simplified audio processing:
   - Removed `TTS` and `faster-whisper` dependencies
   - Added placeholder implementations for speech synthesis and transcription

3. Enhanced PDF processing:
   - Kept `PyMuPDF==1.21.1` for primary PDF processing
   - Added `pypdf` and `pdfplumber` as alternative options if needed

4. Switched from OpenAI to Google Gemini:
   - Now using Google's Gemini Pro model instead of OpenAI
   - Requires a Google API key in the `.env` file as `API_KEY`

## API Key Setup

For the application to work properly, you need to set up a Google API key:

1. **Local Development**: 
   - Create a `.env` file in this directory
   - Add your Google API key: `API_KEY=your_google_api_key`

2. **Hugging Face Deployment**:
   - Add the API_KEY as a repository secret in your Hugging Face Space settings
   - Or set it as an environment variable in the Space settings

## Hugging Face Spaces Deployment

This app is configured for Hugging Face Spaces deployment:

1. The `app.py` file exposes a variable named `demo` that contains the Gradio interface
2. The `.huggingface-space` file contains the configuration for the Space
3. Make sure to set the API_KEY in the Space's environment variables

## Endpoints

### 1. Status Endpoint

- **GET** `/status`
- Returns the status of the API.

### 2. Process Audio

- **POST** `/process-audio`
- Accepts an audio file and returns a synthesized audio response based on the transcribed text.

### 3. Submit Context

- **POST** `/submit-context`
- Accepts job role, job description, and a resume file, returning the processed resume text.

### 4. Generate Interview

- **POST** `/generate-interview`
- Accepts user input and job context, returning an adaptive interview question.

## Requirements

- Gradio
- FastAPI
- Other dependencies listed in `requirements.txt`

## Running the App

To run the app locally, use:

```
python app.py
```

## Deployment

This app is intended to be deployed on Hugging Face Spaces. Ensure all dependencies are listed in `requirements.txt`.
