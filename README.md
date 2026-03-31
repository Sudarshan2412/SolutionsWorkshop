# LangChain Workshop — Handbook + Cats and Dogs Assistant

A hands-on introductory workshop that teaches LangChain by building a multi-source RAG assistant. It can answer questions about your college handbook and the included cats-and-dogs guide, with optional image classification support for cat/dog photos.

---

## Setup

For full setup (including prerequisites and VS Code extensions), see `SETUP_STEPS.md`.

**1. Clone the repo**
```bash
git clone https://github.com/Sudarshan2412/SolutionsWorkshop.git
cd SolutionsWorkshop
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your Groq API key**
- Get a key from [console.groq.com/keys](https://console.groq.com/keys)
- Copy `.env.example` to `.env`
- Paste your key in `.env`

```bash
cp .env.example .env
# then open .env and add: GROQ_API_KEY=your_key_here
```

**5. Prepare your knowledge PDFs**

This workshop uses two sources by default:
- `handbook.pdf` (college policies)
- `cats_and_dogs_notebook.pdf` (pet-care knowledge)

You can replace `handbook.pdf` with your own college handbook. Keep the filename the same so the starter code continues to work.

**6. Early prep for Google Colab (upload only)**

- Open [Google Colab](https://colab.research.google.com/)
- Upload `google_solutions.ipynb` from this repo
- Do not change anything in the notebook yet; this is only to prepare early

---

## Workshop Files (complete these in order)

| File | Concept | What you complete |
|---|---|---|
| `1_hello_chain.py` | Chains | Connect a prompt and LLM using `\|` |
| `2_memory.py` | Memory | Create `ConversationBufferMemory` |
| `3_loader.py` | Document loading | Load PDF + split into chunks |
| `4_retriever.py` | Vector search | Build Chroma vector store + retriever |
| `5_rag_chain.py` | RAG | Assemble the full `RetrievalQA` chain |

Run these one-by-one (not simultaneously) after completing each file:
```bash
python 1_hello_chain.py
python 2_memory.py
python 3_loader.py
python 4_retriever.py
python 5_rag_chain.py
```

If a file still has placeholders, it should print a "not set up yet" message. That is expected for participant mode.

---

## Cats and Dogs in This Workshop

The app includes cat/dog support in two ways:
- Text Q&A via RAG from `cats_and_dogs_notebook.pdf` (for questions like feeding, care, and training).
- Optional image classification in `app.py` using `cat_dog_classifier.keras`.

Image classification is optional. If the model or ML backend is unavailable, the app still runs text RAG normally.

## Running the Chat App

Once all 5 files are complete:
```bash
python app.py
```

Open [http://localhost:7860](http://localhost:7860) in your browser.

Try prompts like:
- "What is the minimum attendance required to sit for exams?"
- "What should I feed my cat?"
- "How should I train my dog?"
- Upload a cat/dog image in chat for optional image classification

---

## Session Execution Guide

See `PARTICIPANT_EXECUTION_FLOW.md` for a shareable step-by-step runbook explaining:
- which file to run at each stage
- why that order matters
- what output to expect
