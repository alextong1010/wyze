import React from 'react';
import './Navbar.css';

function Navbar() {
  return (
    <nav className="navbar">
      <div className="container navbar-container">
        <div className="logo">
          <span className="logo-text">HumanDetect</span>
        </div>
        <ul className="nav-links">
          <li><a href="/" className="active">Home</a></li>
          <li><a href="/about">About</a></li>
        </ul>
      </div>
    </nav>
  );
}

export default Navbar; 