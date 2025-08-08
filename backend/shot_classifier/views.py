import os
import json
import numpy as np
import cv2
import mediapipe as mp
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5
)

# Cricket-relevant keypoint indices (MediaPipe Pose has 33 keypoints)
CRICKET_KEYPOINTS = {
    'nose': 0,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28,
}

SHOT_TYPES = ['drive', 'legglance-flick', 'pull', 'sweep']

class ResNet50CricketClassifier(nn.Module):
    def __init__(self, num_classes=4, pretrained=False):
        super(ResNet50CricketClassifier, self).__init__()
        
        # Load ResNet50
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Modify the final layer for our classification task
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)

# Global model variable
model = None

def load_model():
    """Load the trained ResNet50 model"""
    global model
    if model is None:
        model = ResNet50CricketClassifier(num_classes=len(SHOT_TYPES))
        
        # Load the pretrained model weights
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'ml_model', 'best_resnet50_cricket_model.pth')
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            print(f"Loaded ResNet50 model from {model_path}")
        else:
            print(f"Warning: Model file not found at {model_path}")
            print("Using untrained model - please ensure the model file exists")
        
        model.eval()
    return model

def extract_cricket_pose_keypoints(image):
    """Extract cricket-relevant pose keypoints from image for visualization"""
    # Convert PIL image to RGB
    if isinstance(image, Image.Image):
        image_rgb = image.convert('RGB')
        image_np = np.array(image_rgb)
    else:
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get pose landmarks
    results = pose.process(image_np)
    
    if not results.pose_landmarks:
        return None, None
    
    landmarks = results.pose_landmarks.landmark
    
    # Extract cricket-relevant keypoints
    keypoints = []
    keypoint_coords = []
    
    for name, idx in CRICKET_KEYPOINTS.items():
        landmark = landmarks[idx]
        keypoints.append([landmark.x, landmark.y, landmark.visibility])
        keypoint_coords.append([landmark.x, landmark.y])
    
    return np.array(keypoints), np.array(keypoint_coords)

def preprocess_image_for_resnet(image):
    """Preprocess image for ResNet50 classification"""
    # Define the same transforms used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transforms
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)  # Add batch dimension

@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def classify_shot(request):
    """Classify cricket shot from uploaded image using ResNet50"""
    try:
        if 'image' not in request.FILES:
            return Response(
                {'error': 'No image file provided'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        image_file = request.FILES['image']
        
        # Read and process image
        image = Image.open(image_file).convert('RGB')
        
        # Extract pose keypoints for visualization (not for classification)
        keypoints, keypoint_coords = extract_cricket_pose_keypoints(image)
        
        # Preprocess image for ResNet50 classification
        model_input = preprocess_image_for_resnet(image)
        
        # Load model and predict
        model = load_model()
        
        with torch.no_grad():
            output = model(model_input)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Prepare pose keypoints for frontend visualization (if detected)
        keypoints_for_viz = []
        if keypoints is not None and keypoint_coords is not None:
            for name, idx in CRICKET_KEYPOINTS.items():
                if idx < len(keypoint_coords):
                    kp = keypoint_coords[idx]
                    keypoints_for_viz.append({
                        'name': name,
                        'x': float(kp[0]),
                        'y': float(kp[1])
                    })
        
        response_data = {
            'shot_type': SHOT_TYPES[predicted_class],
            'confidence': round(confidence, 4),
            'pose_keypoints': keypoints_for_viz,
            'pose_detected': len(keypoints_for_viz) > 0,
            'all_probabilities': {
                shot_type: round(probabilities[0][i].item(), 4)
                for i, shot_type in enumerate(SHOT_TYPES)
            }
        }
        
        return Response(response_data, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response(
            {'error': f'Error processing image: {str(e)}'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
def health_check(request):
    """Health check endpoint"""
    return Response({'status': 'healthy'}, status=status.HTTP_200_OK)
