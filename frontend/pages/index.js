import { useState, useRef, useEffect } from 'react';
import Head from 'next/head';

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "https://ranjit21-ai-interview-coaching-agent.hf.space"; 

export default function Home() {
  const [userInput, setUserInput] = useState('');
  const [response, setResponse] = useState('');
  const [jobRole, setJobRole] = useState('');
  const [jobDescription, setJobDescription] = useState('');
  const [resume, setResume] = useState(null);
  const [audioFeedback, setAudioFeedback] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('context');
  const [fileName, setFileName] = useState('');
  const audioRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const handleSubmitContext = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    
    const formData = new FormData();
    formData.append('job_role', jobRole);
    formData.append('job_description', jobDescription);
    if (resume) {
      formData.append('resume', resume);
    }

    try {
      const res = await fetch(`${BACKEND_URL}/submit-context`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        console.error("Error submitting job context:", res.statusText);
      } else {
        setActiveTab('interview');
      }
    } catch (error) {
      console.error("Error submitting job context:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmitInterview = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    
    try {
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
    } catch (error) {
      console.error("Error generating interview:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleAudioSubmit = async (audioBlob) => {
    setIsLoading(true);
    
    try {
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
    } catch (error) {
      console.error("Error processing audio:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const startRecording = async () => {
    if (isRecording) return;

    setIsRecording(true);
    audioChunksRef.current = [];

    try {
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
    } catch (error) {
      console.error("Error starting recording:", error);
      setIsRecording(false);
    }
  };

  const stopRecording = () => {
    if (!isRecording) return;

    mediaRecorderRef.current.stop();
    setIsRecording(false);
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setResume(file);
      setFileName(file.name);
    }
  };

  return (
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8">
      <Head>
        <title>AI Interview Coach | Prepare for your next interview</title>
        <meta name="description" content="AI-powered interview coaching to help you prepare for your next job interview" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-4xl md:text-5xl font-bold text-primary-700 mb-4">
            AI Interview Coach
          </h1>
          <p className="text-lg text-neutral-600 max-w-2xl mx-auto">
            Your personal AI-powered interview coach to help you prepare for your next job interview.
          </p>
        </div>

        <div className="bg-white rounded-2xl shadow-card overflow-hidden">
          <div className="flex border-b border-neutral-200">
            <button 
              onClick={() => setActiveTab('context')}
              className={`flex-1 py-4 px-6 text-center font-medium transition-colors duration-200 ${
                activeTab === 'context' 
                  ? 'text-primary-700 border-b-2 border-primary-500' 
                  : 'text-neutral-500 hover:text-primary-600'
              }`}
            >
              1. Job Details
            </button>
            <button 
              onClick={() => setActiveTab('interview')}
              className={`flex-1 py-4 px-6 text-center font-medium transition-colors duration-200 ${
                activeTab === 'interview' 
                  ? 'text-primary-700 border-b-2 border-primary-500' 
                  : 'text-neutral-500 hover:text-primary-600'
              }`}
            >
              2. Interview Practice
            </button>
          </div>

          <div className="p-6 md:p-8">
            {activeTab === 'context' ? (
              <form onSubmit={handleSubmitContext} className="space-y-6">
                <div>
                  <label htmlFor="jobRole" className="block text-sm font-medium text-neutral-700 mb-1">
                    Job Role <span className="text-red-500">*</span>
                  </label>
                  <input 
                    id="jobRole"
                    type="text" 
                    value={jobRole} 
                    onChange={(e) => setJobRole(e.target.value)} 
                    placeholder="e.g. Software Engineer, Product Manager" 
                    className="input-field" 
                    required
                  />
                </div>
                
                <div>
                  <label htmlFor="jobDescription" className="block text-sm font-medium text-neutral-700 mb-1">
                    Job Description
                  </label>
                  <textarea 
                    id="jobDescription"
                    value={jobDescription} 
                    onChange={(e) => setJobDescription(e.target.value)} 
                    placeholder="Paste the job description here..." 
                    className="input-field min-h-[120px]"
                    rows={4}
                  />
                </div>
                
                <div>
                  <label htmlFor="resume" className="block text-sm font-medium text-neutral-700 mb-1">
                    Upload Resume (PDF or DOCX)
                  </label>
                  <input 
                    id="resume"
                    type="file" 
                    onChange={handleFileChange} 
                    accept="application/pdf, .docx" 
                    className="file-input"
                  />
                  {fileName && (
                    <p className="mt-2 text-sm text-primary-600">
                      Selected file: {fileName}
                    </p>
                  )}
                </div>
                
                <div className="pt-4">
                  <button 
                    type="submit" 
                    className="btn-primary w-full"
                    disabled={isLoading || !jobRole}
                  >
                    {isLoading ? (
                      <span className="flex items-center justify-center">
                        <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Processing...
                      </span>
                    ) : (
                      'Submit & Continue'
                    )}
                  </button>
                </div>
              </form>
            ) : (
              <div className="space-y-8">
                {response && (
                  <div className="card bg-primary-50 mb-6">
                    <h3 className="text-lg font-medium text-primary-800 mb-2">Interviewer:</h3>
                    <p className="text-neutral-700">{response}</p>
                  </div>
                )}
                
                <form onSubmit={handleSubmitInterview} className="space-y-4">
                  <div>
                    <label htmlFor="userInput" className="block text-sm font-medium text-neutral-700 mb-1">
                      Your Answer
                    </label>
                    <textarea 
                      id="userInput"
                      value={userInput} 
                      onChange={(e) => setUserInput(e.target.value)} 
                      placeholder="Type your answer here..." 
                      className="input-field min-h-[120px]"
                      rows={4}
                    />
                  </div>
                  
                  <div className="flex flex-col sm:flex-row gap-3">
                    <button 
                      type="submit" 
                      className="btn-primary flex-1"
                      disabled={isLoading || !userInput}
                    >
                      {isLoading && !isRecording ? (
                        <span className="flex items-center justify-center">
                          <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                          </svg>
                          Processing...
                        </span>
                      ) : (
                        'Submit Text Answer'
                      )}
                    </button>
                    
                    <button 
                      type="button" 
                      onClick={isRecording ? stopRecording : startRecording} 
                      className={`flex items-center justify-center ${
                        isRecording ? 'btn-accent' : 'btn-secondary'
                      }`}
                    >
                      {isRecording ? (
                        <>
                          <span className="relative flex h-3 w-3 mr-2">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-3 w-3 bg-red-500"></span>
                          </span>
                          Stop Recording
                        </>
                      ) : (
                        <>
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                          </svg>
                          Record Answer
                        </>
                      )}
                    </button>
                  </div>
                </form>
                
                {audioFeedback && (
                  <div className="mt-6 p-4 bg-secondary-50 rounded-lg">
                    <h3 className="text-lg font-medium text-secondary-800 mb-2">Audio Feedback:</h3>
                    <audio ref={audioRef} controls src={audioFeedback} className="w-full mt-2" />
                  </div>
                )}
                
                <div className="mt-6 text-center">
                  <button 
                    onClick={() => setActiveTab('context')} 
                    className="btn-outline"
                  >
                    Back to Job Details
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
        
        <div className="mt-12 text-center text-sm text-neutral-500">
          <p>AI Interview Coach &copy; {new Date().getFullYear()} | Your personal interview preparation assistant</p>
        </div>
      </div>
    </div>
  );
}
