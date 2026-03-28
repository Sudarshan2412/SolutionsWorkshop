"""
app.py — Multi-Source RAG Assistant
===================================
Enhanced version with optional image classification routing.

This app now:
1. Routes image paths (.jpg, .png, .jpeg) to cat/dog classification (optional)
2. Processes text queries through standard RAG across indexed documents
3. Maintains conversation memory across interactions

Run this file after completing all setup:
    python app.py

Then open http://localhost:7860 in your browser.

NOTE: Image classification requires proper TensorFlow setup.
      On Python 3.13, use onnxruntime or try Python 3.11/3.12 for full ML support.
"""

import gradio as gr
import importlib
from dotenv import load_dotenv
import os
from typing import Any
from PIL import Image
import numpy as np

# Try to import ML frameworks - graceful fallback
CLASSIFIER_AVAILABLE = False
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

get_rag_chain = importlib.import_module("5_rag_chain").get_rag_chain
load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================
PDF_PATHS = ["cats_and_dogs_notebook.pdf", "handbook.pdf"]
MODEL_PATH = "cat_dog_classifier.keras"
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# ============================================================
# GLOBAL STATE
# ============================================================
classifier_model = None
rag_chain = None
conversation_history = []

# ============================================================
# STEP 1: IMAGE CLASSIFICATION (Optional Feature)
# ============================================================

def load_classifier_model():
    """
    Load the pre-trained cat/dog classifier model.
    Returns None if model file not found or ML framework unavailable.
    """
    global classifier_model
    
    if not CLASSIFIER_AVAILABLE:
        print("[WARNING] ML framework not available. Image classification disabled.")
        print("   Try: pip install tensorflow-cpu (on Python 3.11/3.12)")
        print("   Or: pip install onnxruntime\n")
        return None
    
    if not os.path.exists(MODEL_PATH):
        print(f"[WARNING] {MODEL_PATH} not found in current directory.")
        print("Image classification will be disabled.\n")
        return None
    
    try:
        print(f"Loading classifier model from {MODEL_PATH} using {ML_BACKEND}...")
        
        if ML_BACKEND == "TensorFlow":
            classifier_model = tf.keras.models.load_model(MODEL_PATH)
        elif ML_BACKEND == "ONNX Runtime":
            # For ONNX, convert .keras to .onnx first (manual step)
            print("[NOTE] ONNX Runtime requires .onnx format. .keras model conversion needed.")
            return None
        
        print("[SUCCESS] Classifier model loaded successfully.\n")
        return classifier_model
    except Exception as e:
        print(f"[ERROR] Error loading classifier: {e}\n")
        return None


def preprocess_image(image_path: str, target_size: tuple = (150, 150)) -> np.ndarray:
    """
    Preprocess image for the classifier:
    - Load from path
    - Resize to model's expected input shape
    - Normalize pixel values to [0, 1]
    
    Args:
        image_path: Path to the image file
        target_size: Expected input size for the model (default: 150x150 for this cat/dog classifier)
    
    Returns:
        Preprocessed image as numpy array, or None if loading fails
    """
    try:
        # Load and resize image
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        # Add batch dimension (model expects batch input)
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None


def predict_image(image_path: str) -> str:
    """
    Load image, preprocess it, and get classification prediction.
    
    Args:
        image_path: Path to image file (.jpg, .png, .jpeg)
    
    Returns:
        Human-readable classification string (e.g., "This is a cat image")
    """
    if classifier_model is None:
        print("[ERROR] predict_image called but classifier_model is None!")
        return "Classifier not loaded. Image classification unavailable."
    
    if not os.path.exists(image_path):
        print(f"[ERROR] Image file not found: {image_path}")
        return f"Image file not found: {image_path}"
    
    try:
        print(f"[DEBUG] Processing image: {image_path}")
        
        # Preprocess the image
        img_array = preprocess_image(image_path)
        if img_array is None:
            print("[ERROR] Image preprocessing returned None")
            return "Failed to process image."
        
        print(f"[DEBUG] Image preprocessed. Shape: {img_array.shape}")
        
        # Make prediction
        if ML_BACKEND == "TensorFlow":
            print("[DEBUG] Making TensorFlow prediction...")
            prediction = classifier_model.predict(img_array, verbose=0)
        else:
            # ONNX Runtime prediction
            print("[DEBUG] Making ONNX Runtime prediction...")
            input_name = classifier_model.get_inputs()[0].name
            prediction = classifier_model.run(None, {input_name: img_array})
            prediction = prediction[0]
        
        print(f"[DEBUG] Prediction shape: {prediction.shape}, values: {prediction}")
        
        # Assuming binary classification [cat_prob, dog_prob] or similar
        # Adjust logic based on your model's output structure
        if len(prediction[0]) == 2:
            cat_prob, dog_prob = prediction[0]
            if cat_prob > dog_prob:
                confidence = float(cat_prob)
                result = f"This is a **cat image** (confidence: {confidence:.1%})"
            else:
                confidence = float(dog_prob)
                result = f"This is a **dog image** (confidence: {confidence:.1%})"
        else:
            # Single output neuron (e.g., 0=cat, 1=dog)
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


# ============================================================
# STEP 2: ROUTING LOGIC
# ============================================================

def is_image_path(user_input: str) -> bool:
    """
    Detect if user input is an image file path.
    
    Returns True if input ends with .jpg, .jpeg, or .png
    """
    input_lower = user_input.strip().lower()
    return any(input_lower.endswith(ext) for ext in ALLOWED_IMAGE_EXTENSIONS)


def _extract_uploaded_file_path(uploaded_file: Any) -> str | None:
    """
    Extract file path from possible Gradio upload payload shapes.
    """
    if isinstance(uploaded_file, str):
        return uploaded_file

    if isinstance(uploaded_file, dict):
        path = uploaded_file.get("path")
        if isinstance(path, str):
            return path

    file_name = getattr(uploaded_file, "name", None)
    if isinstance(file_name, str):
        return file_name

    return None


def normalise_chat_message(message: str | dict[str, Any]) -> tuple[str | None, bool]:
    """
    Normalise message from plain text or multimodal Gradio payload.

    Returns:
        Tuple of (normalised_input, force_image_routing)
    """
    if isinstance(message, dict):
        files = message.get("files") or []
        for uploaded_file in files:
            file_path = _extract_uploaded_file_path(uploaded_file)
            if file_path:
                return file_path, True

        text = str(message.get("text", "")).strip()
        return (text or None), False

    if isinstance(message, str):
        text = message.strip()
        return (text or None), False

    text = str(message).strip()
    return (text or None), False


def route_user_input(message: str, force_image: bool = False) -> tuple[dict, str]:
    """
    Route user input to either image classifier or RAG chain.
    
    Returns:
        Tuple of (response_dict, input_type) where input_type is 'image' or 'text'
    """
    is_img_path = force_image or is_image_path(message)
    classifier_ready = classifier_model is not None
    
    print(f"[DEBUG] Routing: is_image_path={is_img_path}, classifier_model={classifier_ready}")
    
    if is_img_path:
        image_path = message.strip()

        if not classifier_ready:
            return {
                "response": (
                    "Image classification is currently unavailable because the "
                    "classifier model is not loaded."
                )
            }, "image_unavailable"

        classification_result = predict_image(image_path)
        
        # Store classification in conversation history
        conversation_history.append({
            "type": "image",
            "input": image_path,
            "classification": classification_result
        })
        
        # Augment the RAG prompt with the classification result
        augmented_query = (
            f"The user uploaded an image classified as: {classification_result}. "
            f"Based on the pet information in your knowledge base, provide helpful information about this animal."
        )
        
        rag_response = rag_chain.invoke({"query": augmented_query})
        
        return {
            "classification": classification_result,
            "response": rag_response["result"],
            "sources": rag_response.get("source_documents", [])
        }, "image"
    
    else:
        # Standard text RAG
        conversation_history.append({
            "type": "text",
            "input": message
        })
        
        rag_response = rag_chain.invoke({"query": message})
        
        return {
            "response": rag_response["result"],
            "sources": rag_response.get("source_documents", [])
        }, "text"


# ============================================================
# STEP 3: GRADIO CHAT INTERFACE
# ============================================================

print("=" * 70)
print("INITIALIZING MULTI-SOURCE RAG ASSISTANT")
print("=" * 70 + "\n")

# Try to load classifier (optional)
if CLASSIFIER_AVAILABLE:
    print("Attempting to load image classifier...")
    load_classifier_model()
    print(f"[DEBUG] Classifier loaded: {classifier_model is not None}\n")
else:
    print("[INFO] Image classification unavailable on this system.")
    print("    (RAG functionality will work normally)\n")

print("Initialising RAG chain - this may take a minute on first run...")
print(
    "(The embedding model is being downloaded and the PDFs are being indexed)"
)
print(f"Indexing sources: {', '.join(PDF_PATHS)}\n")

chain = get_rag_chain(PDF_PATHS)
rag_chain = chain

if chain is None:
    print("\n[ERROR] RAG chain is not set up yet.")
    print("Complete the YOUR CODE HERE section in 5_rag_chain.py first, then re-run app.py.\n")


def chat(message: str | dict[str, Any], history: list) -> str:
    """
    Called by Gradio on every user message.
    Routes to either image classifier or RAG chain based on input type.
    """
    if chain is None:
        return "[ERROR] RAG chain not set. Please complete 5_rag_chain.py"
    
    try:
        normalised_message, force_image = normalise_chat_message(message)

        if not normalised_message:
            return "Please enter a text question or upload an image (.jpg/.jpeg/.png)."

        result, input_type = route_user_input(
            normalised_message,
            force_image=force_image,
        )
        
        if input_type == "image":
            # Format response with classification details
            response_text = f"[IMAGE CLASSIFICATION]\n{result['classification']}\n\nInformation:\n{result['response']}"
        elif input_type == "image_unavailable":
            response_text = result["response"]
        else:
            # Standard text response
            response_text = result['response']
        
        return response_text
    
    except Exception as e:
        return f"[ERROR] Error processing request: {str(e)}"


# ============================================================
# GRADIO UI CONFIGURATION
# ============================================================

classifier_status = "[AVAILABLE]" if classifier_model else "[UNAVAILABLE - RAG still works]"

demo = gr.ChatInterface(
    fn=chat,
    multimodal=True,
    title="Knowledge Base RAG Assistant",
    description=(
        "Ask questions across your indexed knowledge sources (handbook + pet guide in this workshop). "
        "Image paths can be provided for optional classification.\n\n"
        f"Image Classification: {classifier_status}\n\n"
        "How to use:\n"
        "- Handbook: 'What is the minimum attendance required to sit for exams?'\n"
        "- General text: 'What should I feed my cat?'\n"
        "- Image upload: attach an image in chat and send\n"
        "- Image path: '/path/to/image.jpg' (classifier must be loaded)"
    ),
    examples=[
        "What is the minimum attendance required to sit for exams?",
        "What is the policy on late assignment submissions?",
        "How is the final grade calculated in the handbook?",
        "What is the process to apply for leave?",
        "What are common health issues for cats?",
        "How should I train my dog?",
        "What's the best diet for dogs?",
    ],
)

if __name__ == "__main__":
    demo.launch()


