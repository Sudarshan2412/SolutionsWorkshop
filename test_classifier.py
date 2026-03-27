"""Quick test to verify classifier loads and routing works"""
import os
from pathlib import Path

# Simulate the app.py classifier loading logic
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

print(f"CLASSIFIER_AVAILABLE: {CLASSIFIER_AVAILABLE}")
print(f"ML_BACKEND: {ML_BACKEND}\n")

MODEL_PATH = "cat_dog_classifier.keras"

def load_classifier_model():
    """Load the pre-trained cat/dog classifier model."""
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


def is_image_path(user_input: str) -> bool:
    """Detect if user input is an image file path."""
    ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
    input_lower = user_input.strip().lower()
    return any(input_lower.endswith(ext) for ext in ALLOWED_IMAGE_EXTENSIONS)


# Load classifier
print("=" * 70)
print("TEST: Loading Classifier")
print("=" * 70 + "\n")

load_classifier_model()

print(f"[DEBUG] After load_classifier_model(): classifier_model = {classifier_model}")
print(f"[DEBUG] classifier_model is not None: {classifier_model is not None}\n")

# Test routing logic
print("=" * 70)
print("TEST: Routing Logic")
print("=" * 70 + "\n")

test_inputs = [
    "cat_image.jpg",
    "dog.png",
    "What is a cat?",
    "C:\\path\\to\\image.jpeg",
]

for test_input in test_inputs:
    is_img = is_image_path(test_input)
    classifier_ready = classifier_model is not None
    would_route_to_image = is_img and classifier_ready
    
    print(f"Input: '{test_input}'")
    print(f"  is_image_path: {is_img}")
    print(f"  classifier_model is not None: {classifier_ready}")
    print(f"  → Would route to IMAGE: {would_route_to_image}")
    print()
