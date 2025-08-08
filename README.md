# Cricket Shot Classification Webapp

A modern web application for classifying cricket shots from images using **ResNet50 deep learning** and pose estimation visualization.

## Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- Kaggle account (for downloading the dataset)

### Automated Setup (Recommended)
```bash
# Run the setup script
./setup.sh
```

### Manual Setup

#### 1. Backend Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup Django
cd backend
python manage.py migrate
```

#### 2. Frontend Setup
```bash
# Install React dependencies
cd frontend
npm install
```

## Features

- **ResNet50 Classification**: Deep learning model with 85-95% accuracy
- **Pose Visualization**: MediaPipe pose estimation with keypoint overlay
- **UI**: React frontend with drag-and-drop file upload
- **Real-time Results**: Immediate shot classification and pose visualization
- **REST API**: Django backend with comprehensive endpoints

## Architecture

### Classification Pipeline:
```
Input Image → ResNet50 → Classification Result
     ↓
Pose Estimation → Keypoints → Visualization Overlay
```

### Key Features:
1. **ResNet50 Classification**: Uses full RGB images (224x224x3)
2. **Pose Visualization**: MediaPipe pose estimation for overlay
3. **Enhanced Results**: Shot type, confidence, probabilities, pose status

## Project Structure

```
CricketShotEstimation/
├── frontend/                    # React application
│   ├── src/
│   │   ├── App.js              # Main application
│   │   └── components/
│   │       ├── ResultsDisplay.js
│   │       └── PoseVisualizer.js
│   └── package.json
├── backend/                     # Django REST API
│   ├── shot_classifier/
│   │   └── views.py            # ResNet50 classification logic
│   └── manage.py
├── ml_model/                    # Model training
│   ├── resnet50_cricket_classifier.py
│   └── best_resnet50_cricket_model.pth
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## Dataset

### Kaggle Cricket Shot Dataset

This project uses the **Cricket Shot Dataset** available on Kaggle:

** Dataset Information:**
- **Source**: [Cricket Shot Dataset on Kaggle](https://www.kaggle.com/datasets/aneesh10/cricket-shot-dataset/)
- **Size**: 4724 images (675MB total)
- **Shot Types**: 4 categories (drive, legglance-flick, pull, sweep)
- **Format**: PNG images with ~1200 images per shot type
- **Content**: Cropped player shots from various angles and lighting conditions

### Dataset Organization

After downloading, organize the dataset as follows:
```
data/
├── drive/           # Drive shots (~1200 images)
├── legglance-flick/ # Leg glance/flick shots (~1200 images)
├── pull/            # Pull shots (~1200 images)
└── sweep/           # Sweep shots (~1200 images)
```

## Model Training

### Step 1: Prepare Dataset
1. **Download the dataset** from Kaggle using the instructions above
2. **Extract and organize** the images into the folder structure shown above
3. **Verify the structure** - each folder should contain ~1200 images

### Step 2: Train ResNet50 Model
```bash
# Navigate to ml_model directory
cd ml_model

# Install training dependencies
pip install torch torchvision torchaudio
pip install opencv-python pillow matplotlib seaborn scikit-learn tqdm

# Train the model
python resnet50_cricket_classifier.py
```

Or use the Jupyter notebook:
```bash
jupyter notebook resnet50_cricket_classifier.ipynb
```

### Step 3: Verify Model
```bash
# Check if model file was created
ls -la best_resnet50_cricket_model.pth
```

## Running the Application

### 1. Start Django Backend
```bash
# Activate virtual environment
source venv/bin/activate

# Start Django server
cd backend
python manage.py runserver
```
Backend will be available at `http://localhost:8000`

### 2. Start React Frontend
```bash
# In a new terminal
cd frontend
npm start
```
Frontend will be available at `http://localhost:3000`

## Usage

1. Open your browser and go to `http://localhost:3000`
2. Drag and drop a cricket image or click to browse
3. Click "Classify Shot" to analyze the image
4. View the results with:
   - Shot type classification
   - Confidence score
   - All shot probabilities
   - Pose keypoint visualization

## API Endpoints

### Classify Shot
```http
POST /api/classify-shot/
Content-Type: multipart/form-data

Body: image file
```

**Response:**
```json
{
  "shot_type": "drive",
  "confidence": 0.9234,
  "pose_keypoints": [
    {"name": "nose", "x": 0.5, "y": 0.3},
    {"name": "left_shoulder", "x": 0.4, "y": 0.4}
  ],
  "pose_detected": true,
  "all_probabilities": {
    "drive": 0.9234,
    "legglance-flick": 0.0456,
    "pull": 0.0234,
    "sweep": 0.0076
  }
}
```

### Health Check
```http
GET /api/health/
```

## Technologies Used

### Frontend
- **React 18.2.0**: Modern UI framework
- **HTML5 Canvas**: Pose visualization
- **CSS3**: Styling and animations

### Backend
- **Django 4.2+**: Web framework
- **Django REST Framework**: API development
- **Django CORS Headers**: Cross-origin requests

### Machine Learning
- **PyTorch**: Deep learning framework
- **ResNet50**: Pretrained CNN architecture
- **MediaPipe**: Pose estimation
- **OpenCV**: Image processing
- **Pillow**: Image manipulation

### Development
- **Python 3.8+**: Backend language
- **Node.js 16+**: Frontend runtime
- **npm**: Package management


#### Dependencies Issues
```bash
# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install all dependencies
pip install -r requirements.txt

# Frontend dependencies
cd frontend
npm install
```

#### CUDA Issues
```bash
# For CPU-only installation
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
