import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import Navbar from './components/Navbar';
import VideoFeed from './components/VideoFeed';
import LightingBar from './components/LightingBar';
import SystemLog from './components/SystemLog';

const MAX_LOG_MESSAGES = 50;

function App() {
  const [lightingLevel, setLightingLevel] = useState(0);
  const [tvStatus, setTvStatus] = useState('unknown');
  const [logMessages, setLogMessages] = useState([]);
  const previousLightingLevelRef = useRef(lightingLevel);
  const previousTvStatusRef = useRef(tvStatus);

  useEffect(() => {
    const currentLightLevel = lightingLevel;
    const previousLightLevel = previousLightingLevelRef.current;
    const currentTv = tvStatus;
    const previousTv = previousTvStatusRef.current;
    const timestamp = new Date().toLocaleTimeString();

    let newLogEntries = [];

    if (previousLightLevel !== currentLightLevel) {
      if (previousLightLevel === 0 && currentLightLevel === 8) {
        newLogEntries.push({ timestamp, text: "💡 Lights turned ON" });
        console.log("Log: Lights ON");
      } else if (previousLightLevel === 8 && currentLightLevel === 0) {
        newLogEntries.push({ timestamp, text: "💡 Lights turned OFF" });
        console.log("Log: Lights OFF");
      }
      previousLightingLevelRef.current = currentLightLevel;
    }

    if (previousTv !== currentTv) {
      if (currentTv === 'on') {
        newLogEntries.push({ timestamp, text: "📺 TV turned ON (simulated)" });
        console.log("Log: TV ON");
      } else if (currentTv === 'off') {
        if (previousTv === 'on') {
          newLogEntries.push({ timestamp, text: "📺 TV turned OFF (simulated)" });
          console.log("Log: TV OFF");
        } else if (previousTv === 'unknown' || previousTv === 'not_detected') {
          newLogEntries.push({ timestamp, text: "📺 TV is OFF" });
          console.log("Log: TV is OFF");
        }
      } else if (currentTv === 'not_detected') {
        newLogEntries.push({ timestamp, text: "📺 No TV detected in this scene." });
        console.log("Log: TV Not Detected");
      } else if (currentTv === 'unknown') {
      }
      previousTvStatusRef.current = currentTv;
    }

    if (newLogEntries.length > 0) {
      setLogMessages(prevMessages => {
        const updatedMessages = [...prevMessages, ...newLogEntries];
        if (updatedMessages.length > MAX_LOG_MESSAGES) {
          return updatedMessages.slice(updatedMessages.length - MAX_LOG_MESSAGES);
        }
        return updatedMessages;
      });
    }

  }, [lightingLevel, tvStatus]);

  const clearLogMessages = () => {
    console.log("Clearing system log messages...");
    setLogMessages([]);
    previousLightingLevelRef.current = 0;
    previousTvStatusRef.current = 'unknown';
  };

  return (
    <div className="app">
      <Navbar />
      <main className="main-content">
        <div className="container">
          <h1>RoomAware</h1>
          <p className="subtitle">Real-time video analysis with YOLOv8</p>
          <VideoFeed
            setLightingLevel={setLightingLevel}
            setTvStatus={setTvStatus}
            clearLogMessages={clearLogMessages}
          />
        </div>
      </main>
      <LightingBar level={lightingLevel} />
      <SystemLog messages={logMessages} />
      <footer className="footer">
        <div className="container">
          <p>&copy; {new Date().getFullYear()} RoomAware</p>
        </div>
      </footer>
    </div>
  );
}

export default App; 