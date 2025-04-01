// src/components/ConvoAIButton.js
import React, { useState, useRef } from 'react';
import './ConvoAIButton.css';

const ConvoAIButton = ({ onLogMessage }) => {
  const [recording, setRecording] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  // Helper function to format the LLM response like our system logs.
  const formatLLMResponse = (data) => {
    let lightLog = "";
    let thermoLog = "";
    
    // Format lights command similar to TV status formatting:
    if (data.lights && data.lights !== "no_command") {
      if (data.lights.toLowerCase() === 'on') {
        lightLog = "💡 Lights turned ON";
      } else if (data.lights.toLowerCase() === 'off') {
        lightLog = "💡 Lights turned OFF";
      } else {
        lightLog = `💡 Lights: ${data.lights}`;
      }
    } else {
      lightLog = "💡 No light command";
    }
    
    // Format thermostat command
    if (data.thermostat && data.thermostat !== "no_command") {
      if (data.thermostat.startsWith("set_to_")) {
        const temp = data.thermostat.replace("set_to_", "");
        thermoLog = `🌡️ Thermostat set to ${temp}`;
      } else {
        thermoLog = `🌡️ Thermostat: ${data.thermostat}`;
      }
    } else {
      thermoLog = "🌡️ No thermostat command";
    }
    
    return `${lightLog}; ${thermoLog}`;
  };

  const startRecording = async () => {
    setStatusMessage("Requesting microphone access...");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];
      mediaRecorderRef.current.start();
      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };
      setRecording(true);
      setStatusMessage("Recording started...");
    } catch (err) {
      console.error("Error accessing microphone", err);
      setStatusMessage("Error accessing microphone");
    }
  };

  const stopRecording = () => {
    if (!mediaRecorderRef.current) return;
    mediaRecorderRef.current.stop();
    setRecording(false);
    setStatusMessage("Recording stopped. Processing audio...");
    mediaRecorderRef.current.onstop = async () => {
      const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.wav');
      try {
        const response = await fetch('/api/process-audio', {
          method: 'POST',
          body: formData,
        });
        const data = await response.json();
        console.log("Structured output from LLM:", data);
        setStatusMessage("Processing complete. Check system log for output.");
        // Format the response nicely using our new function:
        const formattedResponse = formatLLMResponse(data);
        const timestamp = new Date().toLocaleTimeString();
        onLogMessage({ timestamp, text: formattedResponse });
      } catch (err) {
        console.error("Error sending audio to backend", err);
        setStatusMessage("Error processing audio.");
      }
    };
  };

  return (
    <div style={{ margin: '20px' }}>
      <button className='convo-ai-button' onClick={recording ? stopRecording : startRecording}>
        {recording ? 'End Voice Command' : 'Send a Voice Command'}
      </button>
      <p>{statusMessage}</p>
    </div>
  );
};

export default ConvoAIButton;
