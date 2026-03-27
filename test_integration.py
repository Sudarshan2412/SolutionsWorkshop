"""Integration test: Test the full classifier pipeline"""
import sys
sys.path.insert(0, '.')

# Import the functions from app.py (simulate the app startup)
import os
from pathlib import Path
from PIL import Image
import numpy as np

CLASSIFIER_AVAILABLE = False
classifier_model = None
ML_BACKEND = None

try:
    import tensorflow as tf
    CLASSIFIER_AVAILABLE = True
    ML_BACKEND = "TensorFlow"
except ImportError:
    try:
        import onnxruntime as ort
        CLASSIFIER_AVAILABLE = True
        ML_BACKEND = "ONNX Runtime"
    except ImportError:
        ML_BACKEND = None

MODEL_PATH = "cat_dog_classifier.keras"
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def load_classifier_model():
    global classifier_model
    
    if not CLASSIFIER_AVAILABLE:
        print("[WARNING] ML framework not available. Image classification disabled.")
        return None
    
    if not os.path.exists(MODEL_PATH):
        print(f"[WARNING] {MODEL_PATH} not found in current directory.")
        return None
    
    try:
        print(f"Loading classifier model from {MODEL_PATH} using {ML_BACKEND}...")
        
        if ML_BACKEND == "TensorFlow":
            classifier_model = tf.keras.models.load_model(MODEL_PATH)
        
        print("[SUCCESS] Classifier model loaded successfully.\n")
        return classifier_model
    except Exception as e:
        print(f"[ERROR] Error loading classifier: {e}\n")
        return None

def preprocess_image(image_path: str, target_size: tuple = (150, 150)) -> np.ndarray:
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_image(image_path: str) -> str:
    if classifier_model is None:
        return "Classifier not loaded. Image classification unavailable."
    
    if not os.path.exists(image_path):
        return f"Image file not found: {image_path}"
    
    try:
        print(f"[DEBUG] Processing image: {image_path}")
        
        img_array = preprocess_image(image_path)
        if img_array is None:
            return "Failed to process image."
        
        print(f"[DEBUG] Image preprocessed. Shape: {img_array.shape}")
        
        if ML_BACKEND == "TensorFlow":
            print("[DEBUG] Making TensorFlow prediction...")
            prediction = classifier_model.predict(img_array, verbose=0)
        
        print(f"[DEBUG] Prediction shape: {prediction.shape}, values: {prediction}")
        
        if len(prediction[0]) == 2:
            cat_prob, dog_prob = prediction[0]
            if cat_prob > dog_prob:
                confidence = float(cat_prob)
                result = f"This is a **cat image** (confidence: {confidence:.1%})"
            else:
                confidence = float(dog_prob)
                result = f"This is a **dog image** (confidence: {confidence:.1%})"
        else:
            prediction_value = prediction[0][0]
            if prediction_value < 0.5:
                result = f"This is a **cat image** (score: {1 - prediction_value:.1%})"
            else:
                result = f"This is a **dog image** (score: {prediction_value:.1%})"
        
        print(f"[DEBUG] Classification result: {result}")
        return result
    
    except Exception as e:
        print(f"[ERROR] Error during prediction: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return f"Error during prediction: {e}"

def is_image_path(user_input: str) -> bool:
    input_lower = user_input.strip().lower()
    return any(input_lower.endswith(ext) for ext in ALLOWED_IMAGE_EXTENSIONS)

# ============================================================
# RUN TESTS
# ============================================================

print("=" * 70)
print("INTEGRATION TEST: Full Classifier Pipeline")
print("=" * 70 + "\n")

# Step 1: Load classifier
print("STEP 1: Loading classifier...")
load_classifier_model()
print(f"classifier_model loaded: {classifier_model is not None}\n")

# Step 2: Test with test_image.jpg
test_file = "test_image.jpg"
print(f"STEP 2: Testing with {test_file}...")
print(f"is_image_path('{test_file}'): {is_image_path(test_file)}")
print(f"File exists: {os.path.exists(test_file)}\n")

# Step 3: Attempt classification
print("STEP 3: Running prediction...")
result = predict_image(test_file)
print(f"Result: {result}\n")

# Step 4: Test routing logic
print("STEP 4: Testing routing...")
is_img = is_image_path(test_file)
classifier_ready = classifier_model is not None
would_route = is_img and classifier_ready
print(f"is_image_path: {is_img}")
print(f"classifier_model is not None: {classifier_ready}")
print(f"Would route to image classifier: {would_route}\n")

print("=" * 70)
print("INTEGRATION TEST: COMPLETE")
print("=" * 70)
