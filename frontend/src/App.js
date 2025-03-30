import React from 'react';
import './App.css';
import Navbar from './components/Navbar';
import VideoFeed from './components/VideoFeed';

function App() {
  return (
    <div className="app">
      <Navbar />
      <main className="main-content">
        <div className="container">
          <h1>Human Detection System</h1>
          <p className="subtitle">Real-time video analysis with YOLOv8</p>
          <VideoFeed />
        </div>
      </main>
      <footer className="footer">
        <div className="container">
          <p>&copy; {new Date().getFullYear()} Human Detection System</p>
        </div>
      </footer>
    </div>
  );
}

export default App; 