import React, { useState, useEffect } from 'react';
import './VideoFeed.css';

function VideoFeed() {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  
  const handleImageLoad = () => {
    setIsLoading(false);
    setError(null);
  };
  
  const handleImageError = () => {
    setIsLoading(false);
    setError('Failed to load video stream. Please check if the server is running.');
  };
  
  // Add timestamp to prevent caching
  const videoUrl = `/api/video_feed?t=${new Date().getTime()}`;
  
  return (
    <div className="video-feed-container">
      <div className="video-card">
        <div className="video-header">
          <h2>Live Feed</h2>
          <div className="status-indicator">
            <span className={`status-dot ${error ? 'error' : 'active'}`}></span>
            <span className="status-text">{error ? 'Offline' : 'Online'}</span>
          </div>
        </div>
        
        <div className="video-wrapper">
          {isLoading && (
            <div className="loading-overlay">
              <div className="spinner"></div>
              <p>Loading video stream...</p>
            </div>
          )}
          
          {error && (
            <div className="error-overlay">
              <div className="error-icon">!</div>
              <p>{error}</p>
              <button onClick={() => window.location.reload()}>Retry</button>
            </div>
          )}
          
          <img 
            src={videoUrl} 
            alt="Human Detection Video Stream" 
            onLoad={handleImageLoad}
            onError={handleImageError}
          />
        </div>
        
        <div className="video-info">
          <p>Human detection powered by YOLOv8</p>
          <p className="timestamp">Last updated: {new Date().toLocaleTimeString()}</p>
        </div>
      </div>
    </div>
  );
}

export default VideoFeed; 