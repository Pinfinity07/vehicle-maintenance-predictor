# Vehicle Maintenance Prediction & Agentic Fleet Management

An AI-driven fleet analytics system that predicts vehicle maintenance requirements (Milestone 1) and extends into an agentic AI fleet management assistant (Milestone 2).

**Live Demo (End-Sem):** [Hugging Face Space](https://huggingface.co/spaces/kaori02/vehicle-maintenance-predictor)
**Live Demo (Mid-Sem):** [Hugging Face Space](https://huggingface.co/spaces/kaori02/vehicle-maintenance-predictor)
**GitHub:** [Repository](https://github.com/Pinfinity07/vehicle-maintenance-predictor)

---

## Project Structure

```
vehicle-maintenance-predictor/
├── README.md
├── .gitignore
│
├── # ── Milestone 1: ML Pipeline ──
├── dataset/
│   └── raw_dataset.csv                 # Raw vehicle maintenance dataset
├── scripts/
│   └── train.py                        # Train model & save artifacts
├── app/
│   ├── app.py                          # Mid-sem Gradio web app
│   └── requirements.txt               # Mid-sem dependencies
├── pipeline_modules/
│   ├── __init__.py
│   ├── cleaning.py                     # Data cleaning & preprocessing
│   ├── encoding.py                     # Feature encoding & SMOTE
│   └── training.py                     # Model training with GridSearchCV
├── final_pipeline/
│   └── master_pipeline.ipynb           # Complete ML pipeline notebook
│
├── # ── Milestone 2: Agentic AI ──
└── agentic_app/
    ├── app.py                          # Gradio UI (main entry)
    ├── agent.py                        # LangGraph state-based agent workflow
    ├── model_utils.py                  # ML model training & prediction
    ├── rag_utils.py                    # FAISS RAG pipeline + knowledge base
    ├── requirements.txt                # End-sem dependencies
    └── artifacts/                      # Pre-trained model files
        ├── model.joblib
        ├── preprocessor.joblib
        └── columns.joblib
```

---

## Milestone 1: ML-Based Maintenance Prediction (Mid-Sem)

Classical ML pipeline for predicting vehicle maintenance needs.

### Pipeline

1. **Data Cleaning** — null removal, duplicate handling, date feature engineering, IQR outlier clipping
2. **Feature Engineering** — Mutual Information selection, OrdinalEncoder, OneHotEncoder, RobustScaler
3. **Class Balancing** — SMOTE oversampling on training set
4. **Model Training** — Decision Tree with GridSearchCV (5-fold Stratified CV, F1 optimization)

### Quick Start (Milestone 1)

```bash
pip install -r app/requirements.txt
python scripts/train.py     # Train and save model
python app/app.py            # Launch mid-sem Gradio app
```

---

## Milestone 2: Agentic AI Fleet Management (End-Sem)

LangGraph-based agent that autonomously reasons about vehicle health, retrieves maintenance guidelines via RAG, and generates structured fleet management reports.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Agent Workflow                  │
│                                                             │
│  ┌──────────────┐   ┌──────────────────┐   ┌────────────┐  │
│  │   Analyze     │──▶│     Predict      │──▶│  Retrieve  │  │
│  │   Vehicle     │   │   Maintenance    │   │ Guidelines │  │
│  │              │   │   (ML Model)     │   │  (RAG)     │  │
│  └──────────────┘   └──────────────────┘   └─────┬──────┘  │
│                                                   │         │
│                         ┌─────────────────────────▼──────┐  │
│                         │     Generate Report            │  │
│                         │     (Groq / Llama 3.3 70B)     │  │
│                         └────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Agent Nodes

| Node | Function | Description |
|------|----------|-------------|
| `analyze_vehicle` | Parse & validate | Processes raw vehicle input data |
| `predict_maintenance` | ML prediction | Decision Tree model → risk level + probability |
| `retrieve_guidelines` | RAG retrieval | FAISS + sentence-transformers → relevant maintenance docs |
| `generate_report` | LLM generation | Groq/Llama 3.3 → structured health report |

### Tech Stack

- **Agent Framework:** LangGraph (compiled `StateGraph` with 10 nodes + conditional edges)
- **LLM:** Groq free tier (Llama 3.3 70B Versatile)
- **RAG:** FAISS vector store + `all-MiniLM-L6-v2` embeddings
- **Knowledge Base:** 12 vehicle maintenance guideline documents
- **UI:** Gradio (monochrome theme)
- **ML Model:** Decision Tree + SMOTE + GridSearchCV (from Milestone 1)

### Design Choices & Justification

| Decision | Why |
|---|---|
| **LangGraph over raw LLM chains** | Needs branching (question vs info vs off-topic), state accumulation across turns, and conditional routing between collection / analysis phases. `StateGraph` makes this explicit and debuggable. |
| **Groq Llama 3.3 70B Versatile** | Free tier, fast inference (~500 tok/s), strong instruction-following for JSON outputs needed by the intent classifier and feature extractor. |
| **FAISS + MiniLM-L6-v2** | FAISS is lightweight (no DB server), MiniLM-L6 is 22 MB and produces 384-dim embeddings — enough semantic recall for a ~12-document corpus without GPU. |
| **Two separate RAG corpora** | One for **maintenance guidelines** (prescriptive: "replace brake pads every 50k km") and one for **risk modifiers** (conditional: "rough roads → +20% wear"). Keeping them separate prevents the report generator from mixing prescriptions with risk adjustments. |
| **LLM intent classifier before extraction** | Stops the bot from pushing the same follow-up question when the user is actually asking about options ("what are the owner types"). Avoids the common agentic failure mode of tunnel-visioning on data collection. |
| **Deterministic negation heuristics** | The LLM extractor sometimes misses explicit absence ("no insurance"). A regex fallback guarantees `Insurance_Premium=0` is recorded so the bot stops re-asking. Covered by 15 pytest cases. |
| **Decision Tree (not deep model)** | Trained on 8k synthetic + Kaggle-shape rows; interpretable, fast to train, and `feature_importances_` feeds the "top contributing factors" section of the report. |
| **Rule-based fallback for reports** | Graceful degradation when `GROQ_API_KEY` is absent or the API fails mid-conversation — the app still returns a usable prioritized action plan. |
| **Prompt engineering** | All JSON-producing prompts use explicit schemas + "return ONLY JSON, no markdown" guards, plus a markdown-fence stripper in the parser to handle models that still wrap responses. |

### Structured Output

The agent generates reports containing:
- **Health Summary** — Vehicle status and risk assessment
- **Action Plan** — Prioritized maintenance actions with timelines
- **Maintenance Schedule** — Immediate, 30-day, 90-day, 6-month plan
- **Cost Estimates** — Approximate repair cost ranges
- **Sources** — Retrieved maintenance guidelines that informed the report
- **Disclaimer** — Operational safety notice

### Quick Start (Milestone 2)

```bash
cd agentic_app

# Install dependencies
pip install -r requirements.txt

# Set your Groq API key (copy the example and fill in)
cp .env.example .env
# then edit .env with your key, or export directly:
export GROQ_API_KEY=gsk_your_key_here

# Launch the app
python app.py
```

The app also works **without** a Groq API key — it falls back to a rule-based report generator. With the key, you get full LLM-powered structured reports.

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Optional | Groq API key for LLM report generation. Without it, rule-based fallback is used. See `agentic_app/.env.example`. |

### Running the Tests

```bash
cd agentic_app
pip install pytest
pytest tests/ -v
```

The test suite covers: negation heuristics (15 parameterized cases), intent classification, LangGraph compilation, end-to-end message handling with mocked LLM, and zero-value feature handling. **28 tests, all passing.**

---

## Deployment (Hugging Face Spaces)

The end-sem app is deployed on HuggingFace Spaces:

1. Create a new Space (SDK: Gradio)
2. Upload all files from `agentic_app/`
3. Add `GROQ_API_KEY` as a secret in Space Settings
4. The app auto-deploys

---

## Model Performance

### Milestone 1 (Decision Tree)
- **Accuracy:** 1.0000
- **Precision:** 1.0000
- **Recall:** 1.0000
- **F1-Score:** 1.0000
- **Validation:** 5-fold Stratified Cross-Validation

### Milestone 2 (Agent Quality)
- Correct risk classification across vehicle profiles (CRITICAL → LOW)
- Relevant RAG retrieval (brake query → brake docs, tire query → tire docs)
- Structured LLM output with actionable recommendations
- Graceful fallback when LLM is unavailable

---

## Dependencies

### Milestone 1
scikit-learn, pandas, numpy, imbalanced-learn, gradio, matplotlib

### Milestone 2
langgraph, langchain-groq, langchain-community, langchain-huggingface, faiss-cpu, sentence-transformers, scikit-learn, imbalanced-learn, gradio

---

## License

MIT License
