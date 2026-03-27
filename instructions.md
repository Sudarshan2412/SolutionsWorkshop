# Implementation Plan: Cat/Dog RAG & Image Classifier

## 1. Project Overview
Transition the existing LangChain-based college handbook chatbot into a pet information assistant. 
- **Goal:** Answer text-based questions about cats and dogs using a PDF context.
- **Enhanced Feature:** Integrate a local `.keras` image classifier to identify cats and dogs from user-provided image paths.

---

## 2. Phase 1: Context & RAG Update
- **Target File:** Locate where the `PyPDFLoader` or equivalent is initialized.
- **Action:** - Change the source path from `handbook.pdf` to `cats_and_dogs_notebook.pdf` (or your new filename).
    - Trigger a re-indexing of the Vector Store (ChromaDB, FAISS, etc.) to ensure the new pet data is stored.
- **Prompt for Copilot:** *"Update the PDF loader to use my new 'pets_info.pdf' and clear/rebuild the vector database so it only contains cat and dog information."*

---

## 3. Phase 2: Image Classifier Integration
- **Dependency:** Add `tensorflow` and `Pillow` to requirements.
- **Module Creation:** Create a helper function `predict_image(image_path)` that:
    1. Loads the model using `tensorflow.keras.models.load_model('cat_dog_classifier.keras')`.
    2. Preprocesses the image (resize/normalize) to match the model's input shape.
    3. Returns a string prediction (e.g., "This is a cat").

---

## 4. Phase 3: The "Router" Logic (The 'Brain')
We need to modify the main chat loop to detect if the user input is an image path or a question about an image.

### Logic Flow:
1. **Detect Image:** If the user input ends in `.jpg`, `.png`, or `.jpeg`.
2. **Execute Classifier:** If an image is detected, call `predict_image()`.
3. **Augment Prompt:** Pass the classification result into the LangChain prompt. 
    - *Example:* "The user uploaded an image of a [CAT]. Based on the PDF context, tell them how to care for it."
4. **Fallback:** If no image is detected, proceed with standard RAG retrieval.

---

## 5. Technical Requirements for Copilot
- **Memory Management:** Ensure the `ConversationBufferMemory` persists across both text and image queries.
- **Conditional Chains:** Use `RunnableBranch` or a simple `if/else` block before the chain invokes to decide between `classifier + RAG` or `pure RAG`.
- **Error Handling:** Add a try-except block for `load_model` in case the `.keras` file path is incorrect.