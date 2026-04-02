# Solution Snippets (Paste-Only)

Use this when participants are stuck and need only the exact lines for each `YOUR CODE HERE` block.

## 1_hello_chain.py

```python
chain = prompt | llm
```

## 2_memory.py

```python
memory = ConversationBufferMemory(return_messages=True)
```

## 3_loader.py

Part 1:

```python
loader = PyPDFLoader(pdf_path)
docs = loader.load()
```

Part 2:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
chunks = splitter.split_documents(docs)
```

## 4_retriever.py

```python
vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
```

## 5_rag_chain.py

```python
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
)
```

## Notes

- This file is snippet-only by design.
- Full runnable references stay in the per-file solution scripts.
