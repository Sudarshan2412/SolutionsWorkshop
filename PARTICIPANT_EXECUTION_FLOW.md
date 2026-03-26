# Participant Execution Flow

This guide explains exactly what to run during the workshop, in what order, and why.

## Big Picture

We are building a RAG chatbot in layers. Each file adds one capability:

1. basic chain
2. memory
3. document loading/chunking
4. retrieval
5. full RAG

Running files in order helps you isolate errors and understand each concept before combining them.

## Setup Once

1. Create and activate your virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add your key in `.env`:

```env
GROQ_API_KEY=your_key_here
```

4. Ensure `handbook.pdf` exists in the project root.

## Session Demo Frontend

Before or during the session, you can run the comparison frontend:

```bash
python demo_comparison.py
```

Then open `http://localhost:7860`.

This page shows:
- Workshop guide (run order and flow)
- Live output: without memory vs with memory comparison

## Execution Order (Participant Mode)

Run each file only after filling its `YOUR CODE HERE` section.

```bash
python 1_hello_chain.py
python 2_memory.py
python 3_loader.py
python 4_retriever.py
python 5_rag_chain.py
```

Do not run these 5 files simultaneously in background terminals.

Why:
- The workshop is concept-by-concept, not speed-by-concurrency.
- Sequential runs make debugging easy.
- Output is clearer and easier to explain during a session.

## What To Expect At Each Step

## 1) `1_hello_chain.py` (Chains)
- Goal: connect `prompt | llm`.
- Success signal: it prints a generated explanation for the test topic.
- If incomplete: prints chain-not-set-up message.

## 2) `2_memory.py` (Memory)
- Goal: create `ConversationBufferMemory(return_messages=True)`.
- Success signal: turn 2/3 answers reference earlier conversation.
- If incomplete: prints memory-not-set-up message.

## 3) `3_loader.py` (Loader + Splitter)
- Goal: load PDF and split into chunks.
- Success signal: chunk count plus first chunk preview/metadata.
- If incomplete: prints chunks-not-created message.

## 4) `4_retriever.py` (Vector Store + Retriever)
- Goal: create Chroma vector store and retriever (`k=3`).
- Success signal: top relevant chunks are retrieved for test query.
- If incomplete: prints retriever-not-created message.

## 5) `5_rag_chain.py` (Full RAG)
- Goal: create `RetrievalQA.from_chain_type(...)`.
- Success signal: answer plus source chunks shown.
- If incomplete: prints rag-chain-not-created message.

## Solution Mode (If You Are Stuck or Catching Up)

Run these only when you want to compare with a complete reference:

```bash
python solutions/1_hello_chain_solution.py
python solutions/2_memory_solution.py
python solutions/3_loader_solution.py
python solutions/4_retriever_solution.py
python solutions/5_rag_chain_solution.py
```

Use the matching solution for the file you are currently on.

For quick paste help, use snippet-only answers in:

`solutions/SOLUTION_SNIPPETS.md`

## Final Demo App

After your main files (1-5) are complete, run:

```bash
python app.py
```

Then open `http://localhost:7860`.

`app.py` uses `5_rag_chain.py` from the main folder, not the `solutions/` folder.

Tip: `demo_comparison.py` and `app.py` both use localhost. Stop one before starting the other if port 7860 is already in use.

## Common Mistakes

- Running `app.py` before finishing file 5.
- Running all files in parallel and mixing outputs.
- Editing only solution files and expecting `app.py` to use them.
- Forgetting to add `GROQ_API_KEY` in `.env`.

## Fast Recovery Checklist

1. Confirm current file has no placeholder (`None`) in the part you just completed.
2. Re-run only that file.
3. If still blocked, run matching solution file to compare behavior.
4. Continue in sequence from where you are.
