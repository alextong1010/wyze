import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import Navbar from './components/Navbar';
import VideoFeed from './components/VideoFeed';
import LightingBar from './components/LightingBar';
import SystemLog from './components/SystemLog';
import UserRecognition from './components/UserRecognition';
import io from 'socket.io-client';

// Initialize socket connection (outside component to avoid reconnecting on re-renders)
const socket = io(); // Connects to the same host/port serving the page

import ConvoAIButton from './components/ConvoAIButton';

const MAX_LOG_MESSAGES = 50;
const USER_ABSENCE_TIMEOUT = 2000; // 2 seconds in milliseconds

function App() {
  const [lightingLevel, setLightingLevel] = useState(0);
  const [tvStatus, setTvStatus] = useState('unknown');
  const [logMessages, setLogMessages] = useState([]);
  const [recognizedUser, setRecognizedUser] = useState(null);
  const previousLightingLevelRef = useRef(lightingLevel);
  const previousTvStatusRef = useRef(tvStatus);
  const lightingDebounceTimerRef = useRef(null);
  const userAbsenceTimerRef = useRef(null);

  useEffect(() => {
    const currentLightLevel = lightingLevel;
    const previousLightLevel = previousLightingLevelRef.current;
    const currentTv = tvStatus;
    const previousTv = previousTvStatusRef.current;
    const timestamp = new Date().toLocaleTimeString();

    let newLogEntries = [];

    if (previousLightLevel !== currentLightLevel) {
      if (previousLightLevel === 0 && currentLightLevel > 0) {
        newLogEntries.push({ timestamp, text: `💡 Lights turned ON (Level: ${currentLightLevel}/10)` });
        console.log(`Log: Lights ON (Level: ${currentLightLevel})`);
      } else if (previousLightLevel > 0 && currentLightLevel === 0) {
        newLogEntries.push({ timestamp, text: "💡 Lights turned OFF" });
        console.log("Log: Lights OFF");
      } else if (previousLightLevel !== currentLightLevel) {
        newLogEntries.push({ timestamp, text: `💡 Lighting level changed: ${previousLightLevel} → ${currentLightLevel}` });
        console.log(`Log: Lighting level changed: ${previousLightLevel} → ${currentLightLevel}`);
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

  // Listen for user recognition events and apply preferences
  useEffect(() => {
    const handleUserRecognized = (data) => {
      const timestamp = new Date().toLocaleTimeString();
      
      // Reset user absence timer whenever a user is recognized
      if (userAbsenceTimerRef.current) {
        clearTimeout(userAbsenceTimerRef.current);
      }
      
      // Set new timer
      userAbsenceTimerRef.current = setTimeout(() => {
        if (recognizedUser) {
          // Log user absence
          const absenceTimestamp = new Date().toLocaleTimeString();
          setLogMessages(prevMessages => {
            const newMessage = { 
              timestamp: absenceTimestamp, 
              text: `👤 User ${recognizedUser.name} no longer detected` 
            };
            const updatedMessages = [...prevMessages, newMessage];
            if (updatedMessages.length > MAX_LOG_MESSAGES) {
              return updatedMessages.slice(updatedMessages.length - MAX_LOG_MESSAGES);
            }
            return updatedMessages;
          });
          
          // Clear recognized user
          setRecognizedUser(null);
        }
      }, USER_ABSENCE_TIMEOUT);
      
      // Set the recognized user
      setRecognizedUser(data);
      
      // Apply user preferences
      if (data && data.preferences) {
        // Apply lighting level preference
        if (typeof data.preferences.lighting_level === 'number') {
          setDebouncedLightingLevel(data.preferences.lighting_level);
        }
        
        // Apply TV state preference if TV is detected
        if (data.preferences.tv_state && tvStatus !== 'not_detected') {
          setTvStatus(data.preferences.tv_state);
          // Emit to server to update TV state
          socket.emit('set_tv_state', { state: data.preferences.tv_state });
        }
      }
      
      // Log the recognition event
      setLogMessages(prevMessages => {
        const newMessage = { 
          timestamp, 
          text: `👤 Recognized user: ${data.name} (Preferences applied)` 
        };
        const updatedMessages = [...prevMessages, newMessage];
        if (updatedMessages.length > MAX_LOG_MESSAGES) {
          return updatedMessages.slice(updatedMessages.length - MAX_LOG_MESSAGES);
        }
        return updatedMessages;
      });
    };

    socket.on('user_recognized', handleUserRecognized);

    // Listen for face detection events to reset the timer
    const handleFaceDetected = () => {
      if (recognizedUser && userAbsenceTimerRef.current) {
        // Reset the timer when faces are detected but only if we have a recognized user
        clearTimeout(userAbsenceTimerRef.current);
        userAbsenceTimerRef.current = setTimeout(() => {
          if (recognizedUser) {
            // Log user absence
            const absenceTimestamp = new Date().toLocaleTimeString();
            setLogMessages(prevMessages => {
              const newMessage = { 
                timestamp: absenceTimestamp, 
                text: `👤 User ${recognizedUser.name} no longer detected` 
              };
              const updatedMessages = [...prevMessages, newMessage];
              if (updatedMessages.length > MAX_LOG_MESSAGES) {
                return updatedMessages.slice(updatedMessages.length - MAX_LOG_MESSAGES);
              }
              return updatedMessages;
            });
            
            // Clear recognized user
            setRecognizedUser(null);
          }
        }, USER_ABSENCE_TIMEOUT);
      }
    };

    socket.on('face_detected', handleFaceDetected);

    return () => {
      socket.off('user_recognized', handleUserRecognized);
      socket.off('face_detected', handleFaceDetected);
      if (userAbsenceTimerRef.current) {
        clearTimeout(userAbsenceTimerRef.current);
      }
    };
  }, [tvStatus, recognizedUser]); // Add tvStatus and recognizedUser as dependencies

  // Add this function to reset recognized user
  const resetRecognizedUser = () => {
    if (userAbsenceTimerRef.current) {
      clearTimeout(userAbsenceTimerRef.current);
    }
    setRecognizedUser(null);
  };

  // Modify the lighting level setter with debouncing
  const setDebouncedLightingLevel = (newLevel) => {
    // Clear any existing timer
    if (lightingDebounceTimerRef.current) {
      clearTimeout(lightingDebounceTimerRef.current);
    }
    
    // Set the new level after a short delay (50ms)
    lightingDebounceTimerRef.current = setTimeout(() => {
      setLightingLevel(newLevel);
      lightingDebounceTimerRef.current = null;
    }, 50);
  };

  // Clear the user absence timer when video feed stops or changes
  const handleVideoStateChange = (isPlaying) => {
    if (!isPlaying && userAbsenceTimerRef.current) {
      clearTimeout(userAbsenceTimerRef.current);
      if (recognizedUser) {
        resetRecognizedUser();
      }
    }
  }

  // Callback function for ConvoAIButton to add a new log entry.
  const addLogMessage = (msg) => {
    setLogMessages(prevMessages => {
      const updatedMessages = [...prevMessages, msg];
      if (updatedMessages.length > MAX_LOG_MESSAGES) {
        return updatedMessages.slice(updatedMessages.length - MAX_LOG_MESSAGES);
      }
      return updatedMessages;
    });
  };

  return (
    <div className="app">
      <Navbar />
      <main className="main-content">
        <div className="container">
          <h1>RoomAware</h1>
          <p className="subtitle">Redefining Home AI</p>
          <VideoFeed
            socket={socket}
            setLightingLevel={setDebouncedLightingLevel}
            setTvStatus={setTvStatus}
            clearLogMessages={clearLogMessages}
            resetRecognizedUser={resetRecognizedUser}
            recognizedUser={recognizedUser}
            onVideoStateChange={handleVideoStateChange}
          />
        </div>
      </main>
      <UserRecognition 
        socket={socket} 
        recognizedUser={recognizedUser}
        setRecognizedUser={setRecognizedUser}
      />
      <LightingBar level={lightingLevel} />
      <SystemLog messages={logMessages} />
      <ConvoAIButton onLogMessage={addLogMessage} />
      <footer className="footer">
        <div className="container">
          <p>&copy; {new Date().getFullYear()} RoomAware</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
