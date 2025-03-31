import React, { useState, useEffect, useRef } from 'react';
import io from 'socket.io-client'; // Import socket.io client
import './VideoFeed.css';

// Initialize socket connection (outside component to avoid reconnecting on re-renders)
// Adjust the URL if your Flask app runs on a different port or host
const socket = io(); // Connects to the same host/port serving the page

// Pass setLightingLevel and clearLogMessages down from App
function VideoFeed({ setLightingLevel, setTvStatus, clearLogMessages }) {
  const [isConnected, setIsConnected] = useState(socket.connected);
  const [videoSrc, setVideoSrc] = useState(''); // Store the base64 data URI
  const [currentVideoName, setCurrentVideoName] = useState(''); // Store name from backend status
  const [statusMessage, setStatusMessage] = useState('Connecting...');
  const [error, setError] = useState(null); // Store error messages
  const [availableVideos, setAvailableVideos] = useState([]);
  const [isSwitching, setIsSwitching] = useState(false); // Still useful for button state
  const [isPlaying, setIsPlaying] = useState(false); // New state for play/stop
  const [isReady, setIsReady] = useState(false); // New state, true when backend signals 'ready'

  // Fetch available videos (still needed for the buttons)
  useEffect(() => {
    fetch('/api/videos')
      .then(res => res.ok ? res.json() : Promise.reject('Failed to fetch video list'))
      .then(data => setAvailableVideos(data))
      .catch(err => {
        console.error("Error fetching videos:", err);
        setError('Could not load video list.');
      });
  }, []);

  useEffect(() => {
    // --- Socket Event Listeners ---
    const handleConnect = () => {
      console.log('Socket connected:', socket.id);
      setIsConnected(true);
      setStatusMessage('Connected. Waiting for status...'); // Changed message
    setError(null);
      // Backend will send initial status on connect
    };

    const handleDisconnect = (reason) => {
      console.log('Socket disconnected:', reason);
      setIsConnected(false);
      setStatusMessage('Disconnected.');
      setError('Connection lost.');
      setVideoSrc('');
      setLightingLevel(0);
      setTvStatus('unknown'); // Reset TV status on disconnect
      setIsPlaying(false); // Reset playing state
      setIsReady(false); // Reset ready state
    };

    const handleVideoFrame = (data) => {
      if (data && data.image && typeof data.lightingLevel === 'number') {
        setVideoSrc(`data:image/jpeg;base64,${data.image}`);
        setLightingLevel(data.lightingLevel);
        // Ensure status reflects playing if receiving frames
        if (!isPlaying) setIsPlaying(true);
        if (statusMessage !== 'Online') setStatusMessage('Online');
        if (error) setError(null);
        if (isSwitching) setIsSwitching(false);
      } else {
        console.warn("Received invalid video frame data:", data);
      }
    };

    const handleVideoStatus = (data) => {
      console.log('Video Status:', data);
      setCurrentVideoName(data.video || '');
      setIsSwitching(false); // Stop switching indication on any status update
      setError(null); // Clear errors on status update unless it's an error status

      switch (data.status) {
        case 'playing':
          setStatusMessage('Online');
          setIsPlaying(true);
          setIsReady(false); // Not in ready state anymore
          break;
        case 'ready': // New status
          setStatusMessage(`Ready to play ${data.video || ''}`);
          setIsPlaying(false);
          setIsReady(true); // Set ready state
          setVideoSrc(''); // Clear stale image before play
          break;
        case 'switching':
          setStatusMessage(`Switching to ${data.video}...`);
          setIsSwitching(true);
          setIsPlaying(false); // Stop playing during switch
          setIsReady(false); // Not ready during switch
          setVideoSrc('');
          setLightingLevel(0);
          setTvStatus('unknown'); // Reset TV status during switch
          if (clearLogMessages) {
            clearLogMessages();
          }
          break;
        case 'stopped':
          setStatusMessage(`Stopped. ${data.message || ''}`);
          setIsPlaying(false);
          setIsReady(true); // Usually ready to play again after stop
          // Keep videoSrc as is (last frame) or clear it? Let's clear it.
          // setVideoSrc('');
          setLightingLevel(0);
          setTvStatus('unknown'); // Reset TV status on stop
          break;
        case 'error':
          setStatusMessage(`Error: ${data.message || 'Unknown error'}`);
          setError(data.message || 'An unknown error occurred.');
          setIsPlaying(false);
          setIsReady(false); // Not ready on error
          setVideoSrc('');
          setLightingLevel(0);
          setTvStatus('unknown'); // Reset TV status on error
          break;
        default:
          setStatusMessage('Status update received...');
          setIsPlaying(false); // Default to not playing on unknown status
          setIsReady(false);
      }
    };

    // --- New Listener for TV Status ---
    const handleTvStatusUpdate = (data) => {
        if (data && data.status) {
            console.log('TV Status Update:', data.status);
            setTvStatus(data.status); // Update App state via the passed function
        } else {
            console.warn("Received invalid TV status update:", data);
        }
    };

    // Register listeners
    socket.on('connect', handleConnect);
    socket.on('disconnect', handleDisconnect);
    socket.on('video_frame', handleVideoFrame);
    socket.on('video_status', handleVideoStatus);
    socket.on('tv_status_update', handleTvStatusUpdate); // Register new listener

    // Initial connection check
    if (socket.connected) {
      handleConnect();
    } else {
      socket.connect();
    }

    // Cleanup function: remove listeners when component unmounts
    return () => {
      console.log('Cleaning up socket listeners...');
      socket.off('connect', handleConnect);
      socket.off('disconnect', handleDisconnect);
      socket.off('video_frame', handleVideoFrame);
      socket.off('video_status', handleVideoStatus);
      socket.off('tv_status_update', handleTvStatusUpdate); // Unregister listener
      // Optional: disconnect if the component should fully clean up the connection
      // socket.disconnect();
    };
  }, [setLightingLevel, setTvStatus, clearLogMessages]); // Removed isSwitching dependency

  // --- Button Handlers ---
  const handlePlay = () => {
    if (!isConnected || isPlaying) return;
    console.log("Sending play request...");
    socket.emit('play_request');
    setStatusMessage('Starting...'); // Optimistic UI update
    setIsReady(false); // No longer in ready state once play is requested
  };

  const handleStop = () => {
    if (!isConnected || !isPlaying) return;
    console.log("Sending stop request...");
    socket.emit('stop_request');
    setStatusMessage('Stopping...'); // Optimistic UI update
    setIsPlaying(false); // Assume stop is successful for UI responsiveness
  };

  const handleSwitchVideo = (videoName) => {
    if (isSwitching || videoName === currentVideoName) return;

    console.log(`Requesting switch to video: ${videoName}`);
    setIsSwitching(true);
    setIsPlaying(false); // Stop playing state on switch request
    setIsReady(false); // Not ready during switch
    setStatusMessage(`Requesting switch to ${videoName}...`);
    setError(null);

    fetch('/api/set_video', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ video_name: videoName }),
    })
    .then(res => {
        if (!res.ok) {
        return res.json().then(err => { throw new Error(err.error || `Failed to request switch (${res.status})`) });
        }
        return res.json();
    })
    .then(data => {
      console.log("Switch request successful:", data.message);
      // Wait for 'video_status' event
    })
    .catch(err => {
      console.error("Error requesting video switch:", err);
      setError(err.message || 'Failed to request video switch.');
      setIsSwitching(false); // Reset switching state on error
      setIsReady(false);
      setStatusMessage('Error requesting switch.');
    });
  };

  // Determine status dot class
  const getStatusDotClass = () => {
    if (error) return 'error';
    if (!isConnected) return 'error';
    if (isPlaying) return 'active';
    if (isReady) return 'ready'; // Could add a specific style for 'ready' (e.g., yellow)
    if (isSwitching || statusMessage.includes('Connecting') || statusMessage.includes('Starting') || statusMessage.includes('Stopping')) return 'loading';
    return 'loading'; // Default/fallback
  };

  return (
    <div className="video-feed-container">
      <div className="video-card">
        <div className="video-header">
          <h2>Live Feed {currentVideoName ? `(${currentVideoName})` : ''}</h2>
          <div className="status-indicator">
            <span className={`status-dot ${getStatusDotClass()}`}></span>
            <span className="status-text">{statusMessage}</span>
          </div>
        </div>

        <div className="video-wrapper">
          {/* Loading/Switching Overlay */}
          {(isSwitching || statusMessage.includes('Connecting') || statusMessage.includes('Starting')) && !error && (
            <div className="loading-overlay">
              <div className="spinner"></div>
               <p>{statusMessage}</p>
            </div>
          )}

          {/* Error Overlay */}
          {error && (
            <div className="error-overlay">
              <div className="error-icon">!</div>
              <p>{error}</p>
              {/* Button to try reconnecting socket */}
              {!isConnected && (
                  <button onClick={() => socket.connect()}>Retry Connection</button>
              )}
            </div>
          )}

          {/* Display the image if we have a source and no critical error */}
          {/* Hide image visually if loading/switching */}
            <img
            src={videoSrc}
              alt="Video Stream"
            style={{ display: isConnected && isPlaying && videoSrc && !error ? 'block' : 'none' }}
            // onLoad/onError might not be as relevant for base64 src updates
          />
          {/* Placeholder when ready to play or stopped */}
          {!isPlaying && isReady && !error && (
             <div className="loading-overlay">
                <p>Ready to play.</p>
                <button onClick={handlePlay} className="play-button-overlay">Play</button>
             </div>
          )}

           {/* Placeholder when stopped/disconnected and not ready */}
           {!videoSrc && !isPlaying && !isReady && !error && !isSwitching && (
             <div className="loading-overlay">
                <p>{statusMessage}</p>
                {!isConnected && ( <button onClick={() => socket.connect()}>Retry Connection</button> )}
             </div>
          )}
        </div>

        {/* --- Play/Stop Controls --- */}
        <div className="video-controls play-stop-controls">
           {isConnected && isReady && !isPlaying && (
               <button onClick={handlePlay} disabled={isSwitching}>
                   Play {/* Add Play Icon */}
               </button>
           )}
           {isConnected && isPlaying && (
               <button onClick={handleStop} disabled={isSwitching}>
                   Stop {/* Add Stop Icon */}
               </button>
          )}
        </div>

        <div className="video-controls">
          <h3>Switch Video Source:</h3>
          {availableVideos.length > 0 ? (
            availableVideos.map(videoName => (
              <button
                key={videoName}
                onClick={() => handleSwitchVideo(videoName)}
                // Disable switching if not connected or currently switching
                disabled={isSwitching || !isConnected || videoName === currentVideoName}
                className={videoName === currentVideoName ? 'active' : ''}
              >
                {videoName}
              </button>
            ))
          ) : (
            <p>{error ? 'Could not load video list.' : 'Loading video list...'}</p>
          )}
        </div>

        <div className="video-info">
          <p>RoomAware powered by YOLOv8</p>
          {/* Timestamp could be displayed here if needed, received in socket message */}
        </div>
      </div>
    </div>
  );
}

export default VideoFeed; 