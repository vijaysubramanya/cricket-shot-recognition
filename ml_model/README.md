# Cricket Shot Recognition - ResNet50 Model

This directory contains a deep learning model for cricket shot classification using a ResNet50 architecture with transfer learning. The model can classify cricket shots into four categories: **Drive**, **Leg Glance/Flick**, **Pull**, and **Sweep**.

## Directory Contents

- `resnet50_cricket_classifier.py` - Main training script
- `resnet50_cricket_classifier.ipynb` - Jupyter notebook with interactive training and analysis
- `test_images/` - Sample test images for each shot type
- `resnet_confusion_matrix.png` - Model performance confusion matrix
- `resnet_training_plot.png` - Training curves (loss and accuracy over epochs)
- `resnet50_evaluation_results.json` - Detailed evaluation metrics and classification report
- `resnet50_train_results.json` - Final training epoch results and accuracies

## Model Architecture

### ResNet50 with Transfer Learning
- **Base Model**: Pre-trained ResNet50 (ImageNet weights)
- **Architecture**: 
  - ResNet50 backbone (frozen early layers)
  - Custom classifier head with dropout (0.5)
  - Final layer: 4 classes (shot types)
- **Input Size**: 224×224×3 RGB images
- **Parameters**: ~25M total parameters
- **Training Strategy**: Transfer learning with fine-tuning

### Shot Categories
1. **Drive** - Forward defensive/attacking shots
2. **Leg Glance/Flick** - Shots played to the leg side
3. **Pull** - Horizontal bat shots to short balls
4. **Sweep** - Cross-batted shots against spin

## Training Process

### Data Preprocessing
- **Image Augmentation** (Training):
  - Random horizontal flip (50% probability)
  - Random rotation (±10 degrees)
  - Color jitter (brightness, contrast, saturation, hue)
  - Random affine transformations
  - Normalization (ImageNet stats)

- **Validation/Test**: Resize and normalize only

### Training Configuration
- **Optimizer**: AdamW (lr=0.001, weight_decay=0.01)
- **Loss Function**: Cross-Entropy Loss
- **Scheduler**: ReduceLROnPlateau (patience=5, factor=0.5)
- **Epochs**: 50
- **Batch Size**: Configurable (default: 32)
- **Device**: CUDA (GPU) if available, else CPU

### Data Split
- **Training**: 70%
- **Validation**: 15% 
- **Testing**: 15%

## Model Performance

### Results 
Our ResNet50 model achieved the following metrics on cricket shot classification:

#### Training Results (Final Epoch: 29)
- **Training Accuracy**: 99.5%
- **Validation Accuracy**: 99.8%
- **Training Loss**: 0.02
- **Validation Loss**: 0.03

#### Test Set Performance
- **Overall Accuracy**: 99.0%
- **Total Test Samples**: 945

#### Per-Class Performance
| Shot Type | Precision | Recall | F1-Score | Test Samples |
|-----------|-----------|--------|----------|--------------|
| **Drive** | 98% | 97% | 97% | 245 |
| **Leg Glance/Flick** | 99% | 99% | 99% | 224 |
| **Pull** | 99% | 99% | 99% | 252 |
| **Sweep** | 98% | 100% | 99% | 224 |

#### Aggregate Metrics
- **Macro Average**: Precision: 99%, Recall: 99%, F1-Score: 99%
- **Weighted Average**: Precision: 99%, Recall: 99%, F1-Score: 99%

### Key Achievements
- **Exceeded target accuracy** (99%)
- **Balanced performance** across all shot types
- **Robust classification** with consistent 97-99% metric

### Evaluation Metrics
The model is evaluated using:
- **Accuracy**: Overall classification accuracy
- **Precision, Recall, F1-Score**: Per-class metrics
- **Confusion Matrix**: Detailed classification breakdown
- **Training Curves**: Loss and accuracy progression

### Visualizations
1. **Training Curves** (`resnet_training_results_plot.png`):
   - Training vs Validation Loss
   - Training vs Validation Accuracy
   - Shows model convergence and overfitting detection

2. **Confusion Matrix** (`resnet_confusion_matrix.png`):
   - Actual vs Predicted classifications
   - Identifies which shots are commonly confused

## Usage

### Prerequisites
Before using the model, ensure you have downloaded the pre-trained weights:
1. Download `best_resnet50_cricket_model.pth` from the [Google Drive link](https://drive.google.com/file/d/1JrAERAE0mJsVV4zWzlc7clXwTRQ_qv0p/view?usp=drive_link)
2. Place it in the `backend/ml_model/` directory

### Training the Model
```python
# Run the training script
python resnet50_cricket_classifier.py

# Or use the Jupyter notebook for interactive training
jupyter notebook resnet50_cricket_classifier.ipynb
```

## Requirements

### Dependencies
- PyTorch (with CUDA support recommended)
- torchvision
- scikit-learn
- matplotlib
- seaborn
- PIL (Pillow)
- pandas
- numpy
- opencv-python
- tqdm

### Hardware Recommendations
- **GPU**: NVIDIA GPU with CUDA support (for faster training)
- **RAM**: 8GB+ recommended
- **Storage**: 2GB+ for model and datasets

## Model Files

### Training Generated Files (Included in Repository)
- `resnet50_model_info.json` - Model metadata and configuration
- `resnet50_evaluation_results.json` - Complete evaluation metrics and per-class performance
- `resnet50_train_results.json` - Final training statistics and accuracies
- Training visualization plots (PNG files)

### Pre-trained Model Download
The trained model weights are hosted separately due to file size constraints:

**Download Link**: [ResNet50 Cricket Model - Google Drive](https://drive.google.com/file/d/1JrAERAE0mJsVV4zWzlc7clXwTRQ_qv0p/view?usp=drive_link)

**File Details**:
- **Filename**: `best_resnet50_cricket_model.pth`
- **Size**: ~97 MB
- **Location**: Place in `backend/ml_model/` directory after download

### Setup Instructions
1. Download the model file from the Google Drive link above
2. Place `best_resnet50_cricket_model.pth` in the `backend/ml_model/` directory
3. Ensure the file path matches: `backend/ml_model/best_resnet50_cricket_model.pth`

## Integration

This model is integrated with:
- **Backend API**: Django REST API for serving predictions
- **Frontend**: React application for user interface

## Performance Monitoring

Monitor training progress through:
1. **Console Output**: Real-time epoch results
2. **Training Curves**: Visual loss/accuracy plots
3. **Validation Metrics**: Early stopping based on validation performance
4. **Confusion Matrix**: Detailed classification analysis

## Model Updates

To retrain or fine-tune the model:
1. Update dataset in the appropriate directory structure
2. Modify hyperparameters in the training script
3. Run training with `python resnet50_cricket_classifier.py`
4. Evaluate results using the generated visualizations

## Troubleshooting

### Common Issues
- **Model file not found**: Ensure you've downloaded `best_resnet50_cricket_model.pth` from Google Drive and placed it in `backend/ml_model/`
- **CUDA Out of Memory**: Reduce batch size or use CPU
- **Low Accuracy**: Check data quality, increase epochs, or adjust learning rate
- **Overfitting**: Increase dropout, add regularization, or reduce model complexity

### Performance Tips
- Use GPU for faster training (10-50x speedup)
- Implement early stopping to prevent overfitting
- Use data augmentation to improve generalization
- Monitor validation metrics to tune hyperparameters
