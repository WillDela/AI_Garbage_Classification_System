# Garbage Classification AI

I built this project to see if I could get a CNN to actually tell apart different types of trash, which sounds simple until you realize how similar plastic and metal can look in a photo. The model classifies waste into 12 categories and landed at 83.4% accuracy on the test set, which I was happy with for a first pass, though as I'll get into below, that number hides some categories that really struggled.

## Why 12 categories

I picked these 12 because they're the ones that actually matter for real recycling decisions: battery, biological waste, brown glass, cardboard, clothes, green glass, metal, paper, plastic, shoes, general trash, and white glass. Anything less specific than this and the model isn't really useful for sorting.

## How the model is built

The architecture is a fairly standard CNN setup, but I made a few specific choices worth explaining:

- Three convolutional blocks (32 → 64 → 128 filters), each with batch normalization and a max pool
- Dropout increasing from 0.25 to 0.5 as you go deeper, since I noticed the model overfitting hard without it
- Class weighting, because the dataset isn't evenly split across categories and I didn't want the model just learning to guess "clothes" for everything since it's overrepresented
- Early stopping and a learning rate scheduler so I wasn't just guessing how many epochs to run

Input images are 96x96 RGB, and the whole thing comes out to about 9.7 million parameters.

## Results

Overall test accuracy was 83.4%, but the per-class breakdown tells a more honest story. Clothes and green glass did really well (94.7% and 90.4%), probably because they're visually distinct from everything else in the dataset. Metal and plastic were the weak spots, sitting at 57.4% and 63.8%. I think this is because both categories cover a huge range of shapes and colors — a plastic bottle and a plastic bag don't look anything alike, but they're both labeled "plastic," so the model has a hard time learning one consistent pattern for either category.

I also noticed about an 18% gap between training accuracy (97%) and validation accuracy (82%), which tells me the model is memorizing the training set more than I'd like. That's the main thing I'd want to fix before calling this done.

| Class | Precision | Recall | F1-Score | Accuracy |
|-------|-----------|--------|----------|----------|
| Clothes | 0.94 | 0.95 | 0.94 | 94.7% |
| Green Glass | 0.89 | 0.90 | 0.89 | 90.4% |
| Biological | 0.84 | 0.86 | 0.85 | 85.8% |
| Paper | 0.80 | 0.85 | 0.83 | 85.4% |
| Brown Glass | 0.78 | 0.82 | 0.80 | 82.4% |
| Trash | 0.76 | 0.82 | 0.79 | 81.7% |
| Shoes | 0.77 | 0.81 | 0.79 | 80.8% |
| Cardboard | 0.78 | 0.81 | 0.79 | 80.6% |
| Battery | 0.82 | 0.73 | 0.77 | 72.5% |
| White Glass | 0.75 | 0.67 | 0.71 | 67.2% |
| Plastic | 0.64 | 0.64 | 0.64 | 63.8% |
| Metal | 0.73 | 0.57 | 0.64 | 57.4% |

## Dataset

I used the Kaggle Garbage Classification Dataset (by mostafaabla), which has 15,515 images total. I split it 70/15/15 for train/validation/test, using stratified splitting so each split kept roughly the same class balance as the full dataset. Images got resized to 96x96 using LANCZOS resampling and normalized to a 0-1 range.

## What I'd fix next

- **Data augmentation** — rotating, scaling, and adjusting brightness on training images so the model sees more variation, especially for the categories that are struggling
- **Transfer learning** — trying a pretrained model like ResNet or EfficientNet instead of training from scratch, since 15K images isn't a ton of data for a CNN this size
- **Fixing the overfitting gap** — probably needs stronger regularization or more aggressive augmentation before I trust this model on real-world photos

## Some things worth being honest about

This model was trained on one specific dataset, so it might not generalize well to how trash actually looks in a different region or lighting setup. There's no personal data involved in training, but I'd also want to think about what happens if the model gets something wrong in a real sorting system — misclassifying a battery as general trash isn't just an accuracy statistic, it has an actual environmental consequence.

## Setup

```bash
pip install tensorflow>=2.8.0 numpy>=1.21.0 pandas>=1.3.0 scikit-learn>=1.0.0 matplotlib>=3.5.0 Pillow>=8.0.0 kagglehub
```

```bash
git clone https://github.com/yourusername/garbage-classification-ai.git
cd garbage-classification-ai
pip install -r requirements.txt
```

## Using it

```python
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model('garbage_classifier.h5')

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize((96, 96), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

image_path = "path/to/your/garbage/image.jpg"
processed_img = preprocess_image(image_path)
prediction = model.predict(processed_img)
predicted_class = np.argmax(prediction)

classes = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
           'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']
print(f"Predicted class: {classes[predicted_class]}")
print(f"Confidence: {prediction[0][predicted_class]:.2%}")
```

## License

MIT

## Credit

Dataset from Kaggle (mostafaabla), built with TensorFlow/Keras.
