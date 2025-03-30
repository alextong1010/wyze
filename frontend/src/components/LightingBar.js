import React from 'react';
import './LightingBar.css';
// Optional: Import icons if you want to show them
// import { SunIcon, MoonIcon } from '@heroicons/react/solid'; // Example using heroicons

function LightingBar({ level }) {
  // Ensure level is between 0 and 10
  const clampedLevel = Math.max(0, Math.min(10, level || 0));
  // Calculate width percentage
  const barWidthPercent = `${clampedLevel * 10}%`;

  // Style object to set the CSS variable
  const barStyle = { '--bar-width': barWidthPercent };

  return (
    // Apply the style to the container so the child can use the variable
    <div className="lighting-bar-container" title={`Lighting Level: ${clampedLevel}/10`} style={barStyle}>
      {/* Level div now uses the CSS variable for width */}
      <div className="lighting-bar-level"></div>
      {/* Optional: Add icons instead of scale */}
      {/* <div className="lighting-bar-scale">
        <MoonIcon className="h-5 w-5 text-gray-500" /> // Example icon
        <SunIcon className="h-5 w-5 text-yellow-400" /> // Example icon
      </div> */}
    </div>
  );
}

export default LightingBar; 