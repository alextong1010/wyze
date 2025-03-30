import React, { useEffect, useRef } from 'react';
import './SystemLog.css';

function SystemLog({ messages }) {
  const logEndRef = useRef(null);

  // Auto-scroll to the bottom when messages change
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="system-log-container">
      <h3>System Log</h3>
      <div className="log-messages">
        {messages.length === 0 ? (
          <p className="log-placeholder">No system events yet...</p>
        ) : (
          messages.map((msg, index) => (
            <p key={index} className="log-entry">
              <span className="log-timestamp">[{msg.timestamp}]</span> {msg.text}
            </p>
          ))
        )}
        {/* Dummy element to scroll to */}
        <div ref={logEndRef} />
      </div>
    </div>
  );
}

export default SystemLog; 