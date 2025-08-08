import React, { useState, useRef } from 'react';
import './App.css';
import PoseVisualizer from './components/PoseVisualizer';
import ShotClassifier from './components/ShotClassifier';
import ResultsDisplay from './components/ResultsDisplay';

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileSelect = (file) => {
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setSelectedImage(e.target.result);
        setResults(null);
        setError(null);
      };
      reader.readAsDataURL(file);
    } else {
      setError('Please select a valid image file.');
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    handleFileSelect(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleFileInputChange = (e) => {
    const file = e.target.files[0];
    handleFileSelect(file);
  };

  const handleClassify = async () => {
    if (!selectedImage) return;

    setIsLoading(true);
    setError(null);

    try {
      // Convert data URL to blob
      const response = await fetch(selectedImage);
      const blob = await response.blob();

      // Create form data
      const formData = new FormData();
      formData.append('image', blob, 'cricket_shot.jpg');

      // Send to Django backend
      const apiResponse = await fetch('http://localhost:8000/api/classify-shot/', {
        method: 'POST',
        body: formData,
      });

      if (!apiResponse.ok) {
        throw new Error(`HTTP error! status: ${apiResponse.status}`);
      }

      const data = await apiResponse.json();
      setResults(data);
    } catch (err) {
      setError(`Error classifying image: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedImage(null);
    setResults(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🏏 Cricket Shot Classifier</h1>
        <p>Upload a cricket image to classify the shot type using ResNet50 deep learning</p>
      </header>

      <main className="App-main">
        <div className="upload-section">
          <div
            className="upload-area"
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileInputChange}
              style={{ display: 'none' }}
            />
            {selectedImage ? (
              <div className="image-preview">
                <img src={selectedImage} alt="Selected cricket shot" />
                <button className="reset-btn" onClick={handleReset}>
                  Choose Different Image
                </button>
              </div>
            ) : (
              <div className="upload-placeholder">
                <div className="upload-icon">📁</div>
                <p>Drag and drop an image here, or click to browse</p>
                <p className="upload-hint">Supports: JPG, PNG, GIF</p>
              </div>
            )}
          </div>

          {selectedImage && (
            <div className="action-buttons">
              <button
                className="classify-btn"
                onClick={handleClassify}
                disabled={isLoading}
              >
                {isLoading ? 'Analyzing...' : 'Classify Shot'}
              </button>
            </div>
          )}
        </div>

        {error && (
          <div className="error-message">
            <p>❌ {error}</p>
          </div>
        )}

        {results && (
          <div className="results-section">
            <ResultsDisplay results={results} />
            {results.pose_keypoints && (
              <PoseVisualizer
                image={selectedImage}
                keypoints={results.pose_keypoints}
              />
            )}
          </div>
        )}
      </main>

      <footer className="App-footer">
        <p>Built with React, Django, and PyTorch</p>
      </footer>
    </div>
  );
}

export default App;
