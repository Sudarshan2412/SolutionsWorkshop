# LangChain Workshop — Ask Your College Handbook

A hands-on introductory workshop that teaches LangChain by building a RAG chatbot that answers questions about your college handbook.

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

**5. Add your handbook PDF**

Replace `handbook.pdf` with your college's academic regulations PDF. Keep the filename the same.

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

For complete reference runs, use the matching solution files in `solutions/`.

---

## Running the Chat App

Once all 5 files are complete:
```bash
python app.py
```

Open [http://localhost:7860](http://localhost:7860) in your browser.

---

## Stuck?

Solutions are in the `solutions/` folder. Try for at least 5 minutes before looking!

## Session Execution Guide

See `PARTICIPANT_EXECUTION_FLOW.md` for a shareable step-by-step runbook explaining:
- which file to run at each stage
- why that order matters
- what output to expect
