---

   title: "AI Interview Coaching Agent"

   emoji: "🚀"  

   colorFrom: "indigo"  

   colorTo: "yellow"  

   sdk: "gradio" 

   sdk_version: "5.15.0"  

   app_file: app.py  

   pinned: false

---

# Gradio API for Interview Assistant

This is a Gradio-based API for an interview assistant that processes audio inputs and submits job context information.

## Endpoints

### 1. Status Endpoint

- > **GET** `/status`
- Returns the status of the API.

### 2. Process Audio

- **POST** `/process-audio`
- Accepts an audio file and returns a synthesized audio response based on the transcribed text.

### 3. Submit Context

- **POST** `/submit-context`
- Accepts job role, job description, and a resume file, returning the processed resume text.

## Requirements

- Gradio
- FastAPI
- Other dependencies listed in `requirements.txt`

## Running the App

To run the app locally, use:

```
python backend/main.py
```

## Deployment

This app is intended to be deployed on Hugging Face Spaces. Ensure all dependencies are listed in `requirements.txt`.
