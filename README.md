# AutoML Scientist 🚀 

**AutoML Scientist** is an autonomous AI agent designed to act as your personal Machine Learning Engineer. Built for hackathons and data enthusiasts, it completely automates the machine learning pipeline from raw data upload to a fully trained, exportable model.

## ✨ Features

- **📊 Intelligent Dataset Profiling**: Upload any raw CSV dataset. The system instantly parses it, handles missing values, detects column types, and calculates complex statistical correlations.
- **🎯 Auto Problem Detection**: Automatically detects whether your dataset requires a Classification or Regression model based on the target column cardinality.
- **🧹 AI Code Generation (Data Cleaning)**: Using **Gemini 2.5**, the agent writes custom, dataset-specific Python code to clean your data, handle outliers, encode categories, and impute missing values. It safely sandboxes and executes the code on your data.
- **🔄 Autonomous Research Loop**: Powered by **Groq (Llama-3.3-70b)**, the system orchestrates a multi-step research loop. It recommends different models (Random Forest, XGBoost, Linear Regression, etc.), tunes hyperparameters, executes training scripts in an isolated namespace, and validates accuracy.
- **🏆 Best Model Export**: The orchestrator tracks cross-validation scores across all experiments and saves the winning model to disk using `joblib`.
- **🎨 Glassmorphism UI**: A stunning, modern, responsive frontend that provides a real-time Server-Sent Events (SSE) stream of the AI's internal thoughts and execution logs.

## 🏗️ Architecture

- **Frontend**: Vanilla HTML/JS with a beautiful CSS Glassmorphism aesthetic. Connects to the backend via REST API and SSE (Server-Sent Events) for real-time terminal-like streaming.
- **Backend**: Python Flask REST API following a strict 3-Layer Architecture (Routers → Handlers → Services).
- **LLM Engine**: 
  - `Gemini` handles the raw Python code generation (Journey D).
  - `Groq` acts as the mastermind Orchestrator, deciding which ML pipelines to test (Journey E & F).

## 🚀 How to Run Locally

1. **Clone and Install**
   ```bash
   git clone <your-repo-link>
   cd Hackday/backend
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**
   Create a `.env` file in the `backend/` directory:
   ```env
   GEMINI_API_KEY=your_gemini_key
   GROQ_API_KEY=your_groq_key
   ```

3. **Start the Backend**
   ```bash
   python app.py
   ```
   *Runs on http://127.0.0.1:5000*

4. **Start the Frontend**
   Open a new terminal in the `frontend/` folder:
   ```bash
   python -m http.server 8000
   ```
   *Open http://localhost:8000 in your browser!*

## 📥 Usage

1. Drag and drop any CSV file into the browser window.
2. Click **Analyze Dataset** to view statistics.
3. Click **Start Autonomous Research** to watch the AI stream its thoughts and execute code.
4. Click **Download Best Model (.pkl)** to download the winning trained model ready for deployment!
