import { useState, useRef } from 'react';

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "https://ranjit21-ai-interview-coaching-agent.hf.space"; 

export default function Home() {
  const [userInput, setUserInput] = useState('');
  const [response, setResponse] = useState('');
  const [jobRole, setJobRole] = useState('');
  const [jobDescription, setJobDescription] = useState('');
  const [resume, setResume] = useState(null);
  const [audioFeedback, setAudioFeedback] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const audioRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const handleSubmitContext = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append('job_role', jobRole);
    formData.append('job_description', jobDescription);
    if (resume) {
      formData.append('resume', resume);
    }

    const res = await fetch(`${BACKEND_URL}/submit-context`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      console.error("Error submitting job context:", res.statusText);
    }
  };

  const handleSubmitInterview = async (e) => {
    e.preventDefault();
    const res = await fetch(`${BACKEND_URL}/generate-interview`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ user_input: userInput, job_role: jobRole, job_description: jobDescription }),
    });

    if (!res.ok) {
      console.error("Error generating interview:", res.statusText);
      return;
    }

    const data = await res.json();
    setResponse(data.generated_text);
  };

  const handleAudioSubmit = async (audioBlob) => {
    const formData = new FormData();
    formData.append('audio', audioBlob);

    const res = await fetch(`${BACKEND_URL}/process-audio`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      console.error("Error processing audio:", res.statusText);
      return;
    }

    const data = await res.json();
    setAudioFeedback(data.audio_url);
  };

  const startRecording = async () => {
    if (isRecording) return;

    setIsRecording(true);
    audioChunksRef.current = [];

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorderRef.current = new MediaRecorder(stream);

    mediaRecorderRef.current.ondataavailable = (event) => {
      audioChunksRef.current.push(event.data);
    };

    mediaRecorderRef.current.onstop = async () => {
      const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
      await handleAudioSubmit(audioBlob);
    };

    mediaRecorderRef.current.start();
  };

  const stopRecording = () => {
    if (!isRecording) return;

    mediaRecorderRef.current.stop();
    setIsRecording(false);
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-4">
      <h1 className="text-3xl font-bold mb-6">AI Interview Coach</h1>
      <form onSubmit={handleSubmitContext} className="mb-6 w-full max-w-md bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-xl font-semibold mb-4">Job Details</h2>
        <input type="text" value={jobRole} onChange={(e) => setJobRole(e.target.value)} placeholder="Job Role" className="border border-gray-300 p-2 mb-4 w-full" />
        <input type="text" value={jobDescription} onChange={(e) => setJobDescription(e.target.value)} placeholder="Job Description (optional)" className="border border-gray-300 p-2 mb-4 w-full" />
        <input type="file" onChange={(e) => setResume(e.target.files[0])} accept="application/pdf, .docx" className="mb-4" />
        <button type="submit" className="bg-blue-500 text-white p-2 rounded">Submit Context</button>
      </form>
      <form onSubmit={handleSubmitInterview} className="mb-6 w-full max-w-md bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-xl font-semibold mb-4">Interview Interaction</h2>
        <input type="text" value={userInput} onChange={(e) => setUserInput(e.target.value)} placeholder="Type your answer..." className="border border-gray-300 p-2 mb-4 w-full" />
        <button type="submit" className="bg-blue-500 text-white p-2 rounded">Submit Interview</button>
      </form>
      <button onClick={isRecording ? stopRecording : startRecording} className="bg-green-500 text-white p-2 rounded mb-4">
        {isRecording ? 'Stop Recording' : 'Record Answer'}
      </button>
      {response && <p className="mt-4 text-lg">{response}</p>}
      {audioFeedback && <audio ref={audioRef} controls src={audioFeedback} className="mt-4" />}
    </div>
  );
}
