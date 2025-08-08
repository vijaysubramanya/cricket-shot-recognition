import React from 'react';
import './ResultsDisplay.css';

const ResultsDisplay = ({ results }) => {
  if (!results) return null;

  const { shot_type, confidence, all_probabilities, pose_detected } = results;

  // Get shot emoji based on shot type
  const getShotEmoji = (shotType) => {
    const emojiMap = {
      'drive': '🏏',
      'legglance-flick': '🔄',
      'pull': '⚡',
      'sweep': '🔄'
    };
    return emojiMap[shotType] || '🏏';
  };

  // Get confidence color
  const getConfidenceColor = (conf) => {
    if (conf >= 0.8) return '#28a745';
    if (conf >= 0.6) return '#ffc107';
    return '#dc3545';
  };

  // Sort probabilities for display
  const sortedProbabilities = Object.entries(all_probabilities)
    .sort(([,a], [,b]) => b - a);

  return (
    <div className="results-display">
      <h2>Classification Results</h2>
      
      <div className="main-result">
        <div className="shot-type">
          <span className="shot-emoji">{getShotEmoji(shot_type)}</span>
          <h3>{shot_type.replace('-', ' ').toUpperCase()}</h3>
        </div>
        
        <div className="confidence-meter">
          <div className="confidence-bar">
            <div 
              className="confidence-fill"
              style={{ 
                width: `${confidence * 100}%`,
                backgroundColor: getConfidenceColor(confidence)
              }}
            ></div>
          </div>
          <span className="confidence-text">
            {Math.round(confidence * 100)}% confidence
          </span>
        </div>
      </div>

      <div className="probabilities">
        <h4>All Shot Probabilities</h4>
        <div className="probability-list">
          {sortedProbabilities.map(([shotType, prob]) => (
            <div 
              key={shotType} 
              className={`probability-item ${shotType === shot_type ? 'selected' : ''}`}
            >
              <span className="shot-name">
                {shotType.replace('-', ' ').toUpperCase()}
              </span>
              <div className="probability-bar">
                <div 
                  className="probability-fill"
                  style={{ 
                    width: `${prob * 100}%`,
                    backgroundColor: shotType === shot_type ? getConfidenceColor(prob) : '#e9ecef'
                  }}
                ></div>
              </div>
              <span className="probability-text">
                {Math.round(prob * 100)}%
              </span>
            </div>
          ))}
        </div>
      </div>

      <div className="analysis-tips">
        <h4>Analysis Details</h4>
        <ul>
          <li>✅ ResNet50 model analyzed full image</li>
          <li>✅ Deep learning features extracted automatically</li>
          <li>✅ Robust to lighting and pose variations</li>
          {pose_detected ? (
            <li>✅ Pose keypoints detected for visualization</li>
          ) : (
            <li>ℹ️ No pose detected (classification still works)</li>
          )}
          {confidence < 0.6 && (
            <li>⚠️ Low confidence - consider trying a different image angle</li>
          )}
          {confidence >= 0.9 && (
            <li>🎯 High confidence prediction</li>
          )}
        </ul>
      </div>
    </div>
  );
};

export default ResultsDisplay; 