import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import Navbar from './components/Navbar';
import VideoFeed from './components/VideoFeed';
import LightingBar from './components/LightingBar';
import SystemLog from './components/SystemLog';

const MAX_LOG_MESSAGES = 50;

function App() {
  const [lightingLevel, setLightingLevel] = useState(0);
  const [logMessages, setLogMessages] = useState([]);
  const previousLightingLevelRef = useRef(lightingLevel);

  useEffect(() => {
    const currentLevel = lightingLevel;
    const previousLevel = previousLightingLevelRef.current;
    const timestamp = new Date().toLocaleTimeString();

    let newLogEntry = null;

    if (previousLevel === 0 && currentLevel === 8) {
      newLogEntry = { timestamp, text: "Turning ON the lights!" };
      console.log("Log: Turning ON");
    } else if (previousLevel === 8 && currentLevel === 0) {
      newLogEntry = { timestamp, text: "Turning OFF the lights." };
      console.log("Log: Turning OFF");
    }

    if (newLogEntry) {
      setLogMessages(prevMessages => {
        const updatedMessages = [...prevMessages, newLogEntry];
        if (updatedMessages.length > MAX_LOG_MESSAGES) {
          return updatedMessages.slice(updatedMessages.length - MAX_LOG_MESSAGES);
        }
        return updatedMessages;
      });
    }

    previousLightingLevelRef.current = currentLevel;

  }, [lightingLevel]);

  // Function to clear log messages
  const clearLogMessages = () => {
    console.log("Clearing system log messages...");
    setLogMessages([]);
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