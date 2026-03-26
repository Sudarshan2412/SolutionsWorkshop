# SolutionsWorkshop Setup Guide

This guide helps participants set up the project from scratch.

## 0. Prerequisites

Install these before starting:

1. Python 3.10 or newer
2. Git
3. Visual Studio Code

## 1. Clone the Repository

```bash
git clone https://github.com/Sudarshan2412/SolutionsWorkshop.git
cd SolutionsWorkshop
```

If you already cloned it earlier, pull the latest changes:

```bash
git pull origin main
```

## 2. Create and Activate a Virtual Environment

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Mac/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install Requirements

```bash
pip install -r requirements.txt
```

## 4. Configure Environment Variables

Copy the example env file and add your Groq key.

### Windows (PowerShell)

```powershell
Copy-Item .env.example .env
```

### Mac/Linux

```bash
cp .env.example .env
```

Open .env and set:

```env
GROQ_API_KEY=your_key_here
```

Get your key from: https://console.groq.com/keys

## 5. Required VS Code Extensions

Install these extensions in VS Code:

1. Python — ms-python.python
2. Pylance — ms-python.vscode-pylance

Recommended (optional):

1. Jupyter — ms-toolsai.jupyter
2. GitHub Copilot — github.copilot

## 6. Verify Installation

Run the session demo frontend:

```bash
python demo_comparison.py
```

Open the localhost URL shown in terminal (usually http://localhost:7860).

## 7. Workshop Run Order

Use this order during the session:

```bash
python 1_hello_chain.py
python 2_memory.py
python 3_loader.py
python 4_retriever.py
python 5_rag_chain.py
```

Then run:

```bash
python app.py
```

## 8. If You Get Stuck

- Use snippet-only quick answers: solutions/SOLUTION_SNIPPETS.md
- Use full solution files under the solutions folder
