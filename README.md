# 🗂️ Garbage Classification AI

An intelligent waste classification system using Convolutional Neural Networks (CNNs) to automatically categorize different types of garbage into 12 distinct categories, promoting better waste management and environmental sustainability.

## 📊 Project Overview

This project implements a deep learning solution for garbage classification using TensorFlow and Keras. The model can identify and categorize waste items into 12 different classes, achieving **83.4% accuracy** on test data.

### 🎯 Supported Waste Categories
- Battery
- Biological waste
- Brown glass
- Cardboard  
- Clothes
- Green glass
- Metal
- Paper
- Plastic
- Shoes
- Trash (general)
- White glass

## 🏗️ Model Architecture

The CNN model features a sophisticated architecture designed for optimal image classification:
Input Layer (96×96×3 RGB images)
↓
Conv2D(32) → BatchNorm → Conv2D(32) → MaxPool → Dropout(0.25)
↓
Conv2D(64) → BatchNorm → Conv2D(64) → MaxPool → Dropout(0.25)
↓
Conv2D(128) → BatchNorm → Conv2D(128) → MaxPool → Dropout(0.25)
↓
Flatten → Dense(512) → BatchNorm → Dropout(0.5) → Dense(12, softmax)

**Key Features:**
- **9.7M parameters** for comprehensive feature learning
- **Batch normalization** for training stability
- **Progressive dropout** (0.25 → 0.5) to prevent overfitting  
- **Class weighting** to handle dataset imbalance
- **Early stopping** and **learning rate scheduling** for optimal training

## 📈 Performance Results

| Metric | Score |
|--------|-------|
| **Test Accuracy** | 83.42% |
| **Test Loss** | 0.6612 |
| **Best Validation Accuracy** | 82.55% |
| **Total Parameters** | 9,733,804 |

### 📊 Per-Class Performance

| Class | Precision | Recall | F1-Score | Accuracy |
|-------|-----------|--------|----------|----------|
| **Clothes** | 0.94 | 0.95 | 0.94 | 94.7% |
| **Green Glass** | 0.89 | 0.90 | 0.89 | 90.4% |
| **Biological** | 0.84 | 0.86 | 0.85 | 85.8% |
| **Paper** | 0.80 | 0.85 | 0.83 | 85.4% |
| **Brown Glass** | 0.78 | 0.82 | 0.80 | 82.4% |
| **Trash** | 0.76 | 0.82 | 0.79 | 81.7% |
| **Shoes** | 0.77 | 0.81 | 0.79 | 80.8% |
| **Cardboard** | 0.78 | 0.81 | 0.79 | 80.6% |
| **Battery** | 0.82 | 0.73 | 0.77 | 72.5% |
| **White Glass** | 0.75 | 0.67 | 0.71 | 67.2% |
| **Plastic** | 0.64 | 0.64 | 0.64 | 63.8% |
| **Metal** | 0.73 | 0.57 | 0.64 | 57.4% |

## 🚀 Quick Start

### Prerequisites

```bash
pip install tensorflow>=2.8.0
pip install numpy>=1.21.0
pip install pandas>=1.3.0
pip install scikit-learn>=1.0.0
pip install matplotlib>=3.5.0
pip install Pillow>=8.0.0
pip install kagglehub
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/garbage-classification-ai.git
cd garbage-classification-ai

# Install dependencies
pip install -r requirements.txt
```

### Usage

```python
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load trained model (after training)
model = load_model('garbage_classifier.h5')

# Preprocess image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize((96, 96), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

# Make prediction
image_path = "path/to/your/garbage/image.jpg"
processed_img = preprocess_image(image_path)
prediction = model.predict(processed_img)
predicted_class = np.argmax(prediction)

classes = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
           'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']
print(f"Predicted class: {classes[predicted_class]}")
print(f"Confidence: {prediction[0][predicted_class]:.2%}")
```

## 📁 Dataset Information

- **Source**: Kaggle Garbage Classification Dataset
- **Total Images**: 15,515 images
- **Image Resolution**: 96×96×3 (RGB)
- **Data Split**:
  - Training: 70% (10,860 images)
  - Validation: 15% (2,327 images)
  - Testing: 15% (2,328 images)
- **Class Distribution**: Imbalanced (handled with class weights)

### Data Preprocessing

- Stratified splitting to maintain class balance across splits
- Image normalization (pixel values scaled to 0-1 range)
- High-quality resizing using LANCZOS resampling
- Error handling for corrupted images
- Memory-efficient loading with float32 precision

## 🔧 Technical Implementation

### Key Engineering Decisions

- **Batch Normalization**: Added after each convolutional block for training stability
- **Progressive Dropout**: Increasing dropout rates (0.25 → 0.5) toward output layers
- **Class Weighting**: Computed inverse frequency weights to handle data imbalance
- **Callback Strategy**:
  - Early stopping (patience=7) to prevent overfitting
  - Learning rate reduction (factor=0.5, patience=3) for fine-tuning

### Training Configuration

```python
# Optimizer: Adam (adaptive learning rate)
# Loss: Sparse Categorical Crossentropy
# Metrics: Accuracy
# Batch Size: 32
# Max Epochs: 25 (early stopping enabled)
# Initial Learning Rate: 0.001
```
## 📊 Model Analysis

### Strengths

- ✅ Strong performance on well-represented classes (clothes, glass types)
- ✅ Robust preprocessing pipeline with error handling
- ✅ Comprehensive evaluation with confusion matrix and classification report
- ✅ Addresses class imbalance with weighted training
- ✅ Implements regularization techniques (dropout, batch norm)

### Current Limitations

- ⚠️ Overfitting detected: 18% gap between training (97%) and validation (82%) accuracy
- ⚠️ Poor performance on metal and plastic detection (57-64% accuracy)
- ⚠️ Dataset size limitations (~15K images total)
- ⚠️ Memory constraints limiting larger image resolutions

## 🔮 Future Improvements

### Immediate Enhancements

- **Data Augmentation**: Rotation, scaling, brightness adjustments to increase dataset diversity
- **Advanced Regularization**: L1/L2 regularization, more aggressive dropout
- **Transfer Learning**: Use pre-trained models (ResNet, EfficientNet) for better feature extraction
- **Ensemble Methods**: Combine multiple models for improved accuracy

### Long-term Goals

- **Real-time Classification**: Optimize for mobile deployment
- **Multi-object Detection**: Detect multiple waste items in single image
- **Geographical Adaptation**: Train region-specific models for local waste variations
- **Web Application**: User-friendly interface for waste classification

## 🌍 Environmental Impact & Ethics

### Positive Impact

- **Waste Management**: Automated sorting can improve recycling efficiency
- **Environmental Education**: Helps users understand waste categorization
- **Resource Optimization**: Better sorting leads to improved material recovery

### Ethical Considerations

- **Bias Awareness**: Model trained on specific dataset may not generalize to all regions
- **Privacy**: No personally identifiable information in training data
- **Misclassification Risk**: Incorrect predictions could lead to improper waste handling

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup

```bash
# Fork the repository
# Create a feature branch
git checkout -b feature/amazing-feature

# Make your changes and commit
git commit -m "Add amazing feature"

# Push to the branch
git push origin feature/amazing-feature

# Open a Pull Request
```

## 📄 License

This project is licensed under the MIT License

## 🙏 Acknowledgments

- **Dataset**: Kaggle Garbage Classification Dataset by mostafaabla
- **Framework**: TensorFlow and Keras teams
- **Inspiration**: Environmental sustainability and AI for good initiatives



