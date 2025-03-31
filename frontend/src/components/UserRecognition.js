import React, { useState, useEffect } from 'react';
import './UserRecognition.css';

function UserRecognition({ socket, recognizedUser, setRecognizedUser }) {
  const [availableUsers, setAvailableUsers] = useState([]);
  const [isRegistering, setIsRegistering] = useState(false);
  const [selectedUserId, setSelectedUserId] = useState('');
  const [webcamStream, setWebcamStream] = useState(null);
  const [capturedImage, setCapturedImage] = useState(null);
  const [registrationStatus, setRegistrationStatus] = useState('');
  const [webcamError, setWebcamError] = useState(false);
  const videoRef = React.useRef(null);
  const canvasRef = React.useRef(null);
  const fileInputRef = React.useRef(null);

  // Fetch available users
  useEffect(() => {
    fetch('/api/users')
      .then(res => res.ok ? res.json() : Promise.reject('Failed to fetch users'))
      .then(data => {
        setAvailableUsers(data);
        if (data.length > 0 && !selectedUserId) {
          setSelectedUserId(data[0].id);
        }
      })
      .catch(err => {
        console.error("Error fetching users:", err);
      });
  }, []);

  // Listen for user recognition events
  useEffect(() => {
    if (!socket) return;

    const handleUserRecognized = (data) => {
      console.log('User recognized:', data);
      if (setRecognizedUser) {
        setRecognizedUser(data);
      }
    };

    socket.on('user_recognized', handleUserRecognized);

    return () => {
      socket.off('user_recognized', handleUserRecognized);
    };
  }, [socket, setRecognizedUser]);

  // Handle webcam for face registration
  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setWebcamStream(stream);
      setWebcamError(false);
    } catch (err) {
      console.error("Error accessing webcam:", err);
      setRegistrationStatus('Error: Could not access webcam. You can upload a photo instead.');
      setWebcamError(true);
    }
  };

  const stopWebcam = () => {
    if (webcamStream) {
      webcamStream.getTracks().forEach(track => track.stop());
      setWebcamStream(null);
    }
  };

  const captureImage = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');
      
      // Set canvas dimensions to match video
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      // Draw video frame to canvas
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      // Get data URL from canvas
      const imageData = canvas.toDataURL('image/jpeg');
      setCapturedImage(imageData);
    }
  };

  // Handle file upload
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Check if file is an image
    if (!file.type.match('image.*')) {
      setRegistrationStatus('Error: Please upload an image file (JPEG, PNG, etc.)');
      return;
    }

    // Read the file and convert to data URL
    const reader = new FileReader();
    reader.onload = (e) => {
      setCapturedImage(e.target.result);
      setRegistrationStatus('');
    };
    reader.onerror = () => {
      setRegistrationStatus('Error: Failed to read uploaded file');
    };
    reader.readAsDataURL(file);
  };

  // Trigger file input click
  const triggerFileUpload = () => {
    fileInputRef.current.click();
  };

  const registerFace = async () => {
    if (!capturedImage || !selectedUserId) {
      setRegistrationStatus('Error: No image captured or user selected');
      return;
    }

    setRegistrationStatus('Registering face...');
    
    try {
      const response = await fetch('/api/register_face', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: selectedUserId,
          image_data: capturedImage
        })
      });
      
      const data = await response.json();
      
      if (response.ok) {
        setRegistrationStatus(`Success: ${data.message}`);
        // Close registration panel after successful registration
        setTimeout(() => {
          setIsRegistering(false);
          setCapturedImage(null);
          stopWebcam();
          setWebcamError(false);
        }, 2000);
      } else {
        setRegistrationStatus(`Error: ${data.error}`);
      }
    } catch (err) {
      console.error("Error registering face:", err);
      setRegistrationStatus('Error: Failed to register face');
    }
  };

  const startRegistration = () => {
    setIsRegistering(true);
    setRegistrationStatus('');
    setCapturedImage(null);
    setWebcamError(false);
    startWebcam();
  };

  const cancelRegistration = () => {
    setIsRegistering(false);
    setCapturedImage(null);
    stopWebcam();
    setRegistrationStatus('');
    setWebcamError(false);
  };

  return (
    <div className="user-recognition-container">
      <h3>User Recognition</h3>
      
      {recognizedUser ? (
        <div className="recognized-user">
          <div className="user-avatar">
            <span>{recognizedUser.name.charAt(0)}</span>
          </div>
          <div className="user-info">
            <h4>{recognizedUser.name}</h4>
            <p>Preferences:</p>
            <ul>
              <li>Lighting: {recognizedUser.preferences.lighting_level}/10</li>
              <li>TV: {recognizedUser.preferences.tv_state.toUpperCase()}</li>
            </ul>
          </div>
        </div>
      ) : (
        <p className="no-user">No user recognized</p>
      )}
      
      <button 
        className="register-face-btn" 
        onClick={startRegistration}
      >
        Register Face
      </button>
      
      {isRegistering && (
        <div className="registration-panel">
          <h4>Face Registration</h4>
          
          <div className="user-select">
            <label>Select User:</label>
            <select 
              value={selectedUserId} 
              onChange={(e) => setSelectedUserId(e.target.value)}
            >
              {availableUsers.map(user => (
                <option key={user.id} value={user.id}>{user.name}</option>
              ))}
            </select>
          </div>
          
          <div className="webcam-container">
            {!capturedImage ? (
              webcamError ? (
                <div className="upload-placeholder">
                  <p>Webcam unavailable. Please upload a photo instead.</p>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleFileUpload}
                    style={{ display: 'none' }}
                  />
                  <button 
                    onClick={triggerFileUpload} 
                    className="upload-btn"
                  >
                    Choose Photo
                  </button>
                </div>
              ) : (
                <video 
                  ref={videoRef} 
                  autoPlay 
                  playsInline 
                  muted
                />
              )
            ) : (
              <img src={capturedImage} alt="Captured" />
            )}
            <canvas ref={canvasRef} style={{ display: 'none' }} />
          </div>
          
          <div className="registration-controls">
            {!capturedImage ? (
              webcamError ? (
                <button onClick={cancelRegistration}>Cancel</button>
              ) : (
                <>
                  <button onClick={captureImage}>Capture</button>
                  <button onClick={cancelRegistration}>Cancel</button>
                </>
              )
            ) : (
              <>
                <button onClick={() => {
                  setCapturedImage(null);
                  if (webcamError) {
                    setRegistrationStatus('');
                  }
                }}>
                  {webcamError ? 'Choose Different Photo' : 'Retake'}
                </button>
                <button onClick={registerFace}>Register</button>
                <button onClick={cancelRegistration}>Cancel</button>
              </>
            )}
          </div>
          
          {registrationStatus && (
            <p className={`registration-status ${registrationStatus.includes('Error') ? 'error' : ''}`}>
              {registrationStatus}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

export default UserRecognition; 