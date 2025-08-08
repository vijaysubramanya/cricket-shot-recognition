import React, { useEffect, useRef } from 'react';
import './PoseVisualizer.css';

const PoseVisualizer = ({ image, keypoints }) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!image || !keypoints || keypoints.length === 0) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const img = new Image();

    img.onload = () => {
      // Set canvas size to match image
      canvas.width = img.width;
      canvas.height = img.height;

      // Draw the image
      ctx.drawImage(img, 0, 0);

      // Draw pose keypoints
      drawPoseKeypoints(ctx, keypoints, img.width, img.height);
    };

    img.src = image;
  }, [image, keypoints]);

  const drawPoseKeypoints = (ctx, keypoints, imgWidth, imgHeight) => {
    // Define connections between keypoints for skeleton drawing
    const connections = [
      ['left_shoulder', 'right_shoulder'],
      ['left_shoulder', 'left_elbow'],
      ['right_shoulder', 'right_elbow'],
      ['left_elbow', 'left_wrist'],
      ['right_elbow', 'right_wrist'],
      ['left_shoulder', 'left_hip'],
      ['right_shoulder', 'right_hip'],
      ['left_hip', 'right_hip'],
      ['left_hip', 'left_knee'],
      ['right_hip', 'right_knee'],
      ['left_knee', 'left_ankle'],
      ['right_knee', 'right_ankle'],
    ];

    // Draw connections (skeleton)
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';

    connections.forEach(([start, end]) => {
      const startPoint = keypoints.find(kp => kp.name === start);
      const endPoint = keypoints.find(kp => kp.name === end);

      if (startPoint && endPoint) {
        const x1 = startPoint.x * imgWidth;
        const y1 = startPoint.y * imgHeight;
        const x2 = endPoint.x * imgWidth;
        const y2 = endPoint.y * imgHeight;

        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }
    });

    // Draw keypoints
    keypoints.forEach(keypoint => {
      const x = keypoint.x * imgWidth;
      const y = keypoint.y * imgHeight;

      // Draw keypoint circle
      ctx.fillStyle = '#ff0000';
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, 2 * Math.PI);
      ctx.fill();

      // Draw keypoint border
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, 2 * Math.PI);
      ctx.stroke();

      // Draw keypoint label
      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = '#000000';
      ctx.lineWidth = 1;
      ctx.font = '12px Arial';
      ctx.textAlign = 'center';
      
      const label = keypoint.name.replace('_', ' ');
      const textWidth = ctx.measureText(label).width;
      
      // Draw text background
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.fillRect(x - textWidth/2 - 2, y - 20, textWidth + 4, 16);
      
      // Draw text
      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, x, y - 8);
    });
  };

  if (!image || !keypoints) return null;

  return (
    <div className="pose-visualizer">
      <h3>Pose Detection Visualization</h3>
      <div className="canvas-container">
        <canvas
          ref={canvasRef}
          className="pose-canvas"
          style={{ maxWidth: '100%', height: 'auto' }}
        />
      </div>
      <div className="pose-info">
        <p>
          <strong>Detected Keypoints:</strong> {keypoints.length} points
        </p>
        <p>
          <strong>Key Areas:</strong> Shoulders, Elbows, Wrists, Hips, Knees, Ankles
        </p>
      </div>
    </div>
  );
};

export default PoseVisualizer; 