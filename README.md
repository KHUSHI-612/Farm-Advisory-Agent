# CropCast — Intelligent Farm Advisory Agent

**End-to-End Crop Yield Prediction + Agentic AI Farm Advisory System**

A modern agricultural intelligence platform combining machine learning yield prediction with a conversational AI agronomist. Built with **Random Forest**, **LangGraph**, **RAG**, and **Claude API**.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://shiavm006-farm-advisory-agent-streamlit-app-e7txgg.streamlit.app/)

## Features

### **Milestone 1: ML Yield Prediction**
- **Random Forest Regressor** trained on FAO global crop production data
- Supports **8 major crops** across **25+ countries**
- Real-time prediction from climate + soil inputs
- Interactive **Plotly visualizations** (gauge charts, radar plots, feature importance)
- Professional **PDF report generation**

### **Milestone 2: Agentic AI Advisory**
- **Conversational agronomist** powered by **Claude AI + LangGraph**
- **Multi-turn chat** with conversation memory
- **Document upload** (PDF/TXT) — upload field reports, soil tests, research papers
- **RAG knowledge base** with agronomy guidelines from FAO, extension services
- **Structured recommendations** — yield drivers, pest management, irrigation timing
- **Retrieval-augmented reasoning** — agent cites sources for every recommendation

## Quick Start

### Local Development

1. **Clone & setup**
   ```bash
   git clone <your-repo>
   cd Farm-Advisory-Agent
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Add API key** (for conversational agent)
   ```bash
   echo "ANTHROPIC_API_KEY=sk-ant-a" > .env
   ```

3. **Run the app**
   ```bash
   streamlit run streamlit_app.py
   ```
   
   Open http://localhost:8501

### Cloud Deploy (Streamlit Cloud)

1. **Push to GitHub** and deploy on [Streamlit Cloud](https://share.streamlit.io)
2. **Add secrets** in your Streamlit app dashboard:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-api03-..."
   ```
3. **Done!** — No external databases or infrastructure needed.

## How to Use

### 1. **Yield Prediction** (Predict Yield tab)
- Set **farm parameters**: crop, region, rainfall, temperature, pesticides
- Click **"Run Prediction"** → get yield forecast + quality band + risk level
- Download **professional PDF report** with charts and analysis

### 2. **Conversational Agent** (Advisory Agent tab)
- **Set farm context** (crop, region, climate) in the collapsible panel
- **Upload documents** (optional) — field reports, soil tests, research papers  
- **Chat naturally**: *"What should I watch for this season?"* / *"How do I improve yield?"*
- Get **detailed responses** with sources, structured recommendations, and safety notes

### 3. **Model Insights** (Insights tab)
- Feature importance analysis
- Performance metrics (R² ~0.87 on test set)
- Training configuration details

## Technical Architecture

### **ML Pipeline**
- **Model**: Random Forest Regressor (100 estimators)
- **Features**: Year, Rainfall, Temperature, Pesticides + One-hot encoded Crop + Area
- **Target**: FAO yield data (hg/ha → t/ha conversion)
- **Preprocessing**: StandardScaler, missing value imputation
- **Hosting**: HuggingFace Hub ([shiavm006/Crop-yield_pridiction](https://huggingface.co/shiavm006/Crop-yield_pridiction))

### **Agentic AI System**
- **Framework**: LangGraph (state-based workflow)
- **LLM**: Claude (Anthropic) with model fallbacks
- **RAG**: FAISS vector store + HuggingFace embeddings
- **Knowledge Base**: FAO crop guidelines, extension service docs (in `rag/docs/`)
- **Anti-hallucination**: Grounded retrieval, source attribution, structured output

### **UI/UX**
- **Framework**: Streamlit with custom CSS theming
- **Design**: Dark/light mode, responsive layout, professional styling
- **Charts**: Plotly (interactive gauges, radar plots, bar charts)
- **Reports**: ReportLab PDF generation

## Project Structure

```
Farm-Advisory-Agent/
├── streamlit_app.py             # Main Streamlit application
├── report_generator.py          # PDF report generation
├── requirements.txt             # Python dependencies
├── agent/                       # Agentic AI system
│   ├── graph.py                 #   LangGraph workflow (run_agent, run_chat)
│   ├── nodes.py                 #   Individual agent nodes (predict, retrieve, advise)
│   ├── state.py                 #   Shared state schema (FarmState)
│   └── prompts.py               #   LLM prompts + anti-hallucination rules
├── rag/                         # Retrieval-Augmented Generation
│   ├── ingest.py                #   Build FAISS vector store from docs/
│   ├── retriever.py             #   Query vector store for relevant chunks
│   ├── docs/                    #   Agronomy knowledge base (.txt files)
│   └── vectorstore/             #   FAISS index (auto-generated)
├── model/                       # ML model artifacts (optional local storage)
├── Dataset/                     # Training data (yield_df.csv)
└── report/                      # LaTeX report source
```

## Key Capabilities

| Feature | Milestone 1 | Milestone 2 |
|---------|-------------|-------------|
| **ML Prediction** | Yes - Random Forest yield forecasting | Yes - Integrated into agent reasoning |
| **Data Sources** | Yes - FAO global crop production | Yes - + Agronomy guidelines via RAG |
| **User Interface** | Yes - Form-based input/output | Yes - + Conversational chat interface |
| **Report Generation** | Yes - PDF reports with charts | Yes - + Structured agent recommendations |
| **Personalization** | Yes - Farm-specific predictions | Yes - + Multi-turn conversation memory |
| **Knowledge Integration** | No - Static model only | Yes - Dynamic retrieval + reasoning |
| **Document Processing** | No - No file uploads | Yes - PDF/TXT upload + analysis |
| **Deployment** | Yes - Self-contained Streamlit app | Yes - + API key configuration |

## Model Performance

- **Training R²**: ~0.94
- **Test R²**: ~0.87  
- **Features**: Rainfall (28%), Temperature (22%), Pesticides (19%), Year (16%), Crop type (15%)
- **Evaluation**: MAE, RMSE, R² metrics across crop types and regions
- **Validation**: Holds out recent years for temporal validation

## Supported Crops & Regions

**Crops**: Wheat, Rice (paddy), Maize, Potatoes, Cassava, Soybeans, Sorghum, Sweet Potatoes

**Regions**: India, Brazil, USA, China, Argentina, Australia, Canada, France, Germany, Indonesia, Mexico, Nigeria, Pakistan, Russia, Thailand, Turkey, Ukraine, and more.

## 🔧 Environment Variables

Create a `.env` file:
```bash
# Required for conversational agent
ANTHROPIC_API_KEY=sk-ant-api03-...

# Optional: specify preferred Claude model
ANTHROPIC_MODEL=claude-haiku-4-5
```

## Troubleshooting

### **"No module named 'langchain_huggingface'"**
- Ensure `langchain-huggingface` is in `requirements.txt`
- Reinstall: `pip install -r requirements.txt`

### **"ANTHROPIC_API_KEY not set"**
- Add API key to `.env` file or Streamlit secrets
- Restart the app after adding the key

### **FAISS build fails**
- Check that `rag/docs/` contains `.txt` files
- Run manually: `python -m rag.ingest`

### **Model download fails**
- Requires internet connection to download from HuggingFace
- Alternatively, place `model.pkl`, `scaler.pkl`, `features.pkl` in `model/` folder

## Academic Context

This project demonstrates advanced ML engineering and agentic AI concepts:

- **Feature Engineering**: Temporal, categorical, and numerical feature processing
- **Model Deployment**: HuggingFace Hub integration, caching strategies  
- **Agentic Workflows**: State-based reasoning with LangGraph
- **Retrieval-Augmented Generation**: Vector search + LLM reasoning
- **Anti-Hallucination**: Source attribution, structured output, fallback systems
- **Production UI/UX**: Professional Streamlit application with custom styling

## License

MIT License — free for academic and commercial use.

## Contributing

Contributions welcome! Areas for improvement:
- Additional crop types and regions
- Weather API integration for real-time forecasting  
- Soil sensor data integration
- Multi-language support for global deployment
- Advanced visualizations (satellite imagery, field mapping)

---

**Built with**: Python • Streamlit • LangGraph • Claude AI • FAISS • Random Forest • Plotly • ReportLab