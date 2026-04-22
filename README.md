# CropCast — Intelligent Farm Advisory Agent

**End-to-End Crop Yield Prediction + Agentic AI Farm Advisory System**

A modern agricultural intelligence platform combining machine learning yield prediction with a conversational AI agronomist. Built with **Random Forest**, **LangGraph**, **RAG**, and **Claude API**.

[Open in Streamlit](https://shiavm006-farm-advisory-agent-streamlit-app-e7txgg.streamlit.app/)

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
   Open [http://localhost:8501](http://localhost:8501)

### Cloud Deploy (Streamlit Cloud)

1. **Push to GitHub** and deploy on [Streamlit Cloud](https://share.streamlit.io)
2. **Add secrets** in your Streamlit app dashboard:
  ```toml
   ANTHROPIC_API_KEY = ""
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

## Technical Architecture & Design Decisions

### **ML Pipeline**

- **Model**: Random Forest Regressor (100 estimators)
  - **Why Random Forest**: Handles mixed data types (numerical + categorical), robust to outliers, provides feature importance, works well with limited data (~1000 samples), naturally handles missing values, and interpretable for agricultural stakeholders
  - **Why not Deep Learning**: Small dataset size, need for interpretability in agriculture, computational efficiency for real-time prediction, and avoiding overfitting with limited training data
- **Features**: Year, Rainfall, Temperature, Pesticides + One-hot encoded Crop + Area
  - **Feature Selection Rationale**: Based on agricultural science - climate (rainfall, temperature) drives photosynthesis, pesticides affect crop health, temporal trends (year) capture improved varieties/practices, crop type determines growth patterns, region captures soil/climate conditions
- **Target**: FAO yield data (hg/ha → t/ha conversion)
  - **Why FAO Data**: Standardized global dataset, consistent methodology across countries, covers major food crops, reliable institutional source with quality controls
- **Preprocessing**: StandardScaler, missing value imputation
  - **Why StandardScaler**: Random Forest doesn't require scaling, but enables fair feature importance comparison and prepares for potential model ensemble
- **Hosting**: HuggingFace Hub ([shiavm006/Crop-yield_pridiction](https://huggingface.co/shiavm006/Crop-yield_pridiction))
  - **Why HuggingFace**: Version control for ML artifacts, free hosting, seamless integration with deployment, reproducibility, and community accessibility

### **Agentic AI System**

- **Framework**: LangGraph (state-based workflow)
  - **Why LangGraph**: Explicit state management for complex agricultural reasoning, graph-based flow control, built for production LLM applications, better than simple prompt chains for multi-step advisory generation
  - **Why not LangChain alone**: Need explicit state passing between nodes (farm context → ML prediction → risk assessment → retrieval → advice), complex conditional logic, and workflow visualization
- **LLM**: Claude (Anthropic) with model fallbacks
  - **Why Claude**: Strong reasoning capabilities for agricultural analysis, good at structured output (JSON), safety-focused (important for agricultural advice), longer context window for document processing
  - **Why Model Fallbacks**: Different users have different API access tiers, ensures system reliability, graceful degradation to local templates if all models fail
- **RAG**: FAISS vector store + HuggingFace embeddings
  - **Why FAISS**: Fast similarity search, works offline, no external database required, handles local deployment, good for < 10K documents
  - **Why HuggingFace Embeddings**: Free, no API keys, consistent with open-source approach, good quality for agricultural text, works offline
  - **Why not Pinecone/Weaviate**: Adds deployment complexity, requires external services, overkill for this document volume
- **Knowledge Base**: FAO crop guidelines, extension service docs (in `rag/docs/`)
  - **Content Strategy**: Authoritative sources (FAO), practical guidance (extension services), covers major crops and regions, text format for easy processing
- **Anti-hallucination**: Grounded retrieval, source attribution, structured output
  - **Critical for Agriculture**: Wrong advice can cause crop loss, need traceable recommendations, structured output enables validation, source attribution builds trust

### **UI/UX Design Philosophy**

- **Framework**: Streamlit with custom CSS theming
  - **Why Streamlit**: Rapid prototyping, Python-native (matches ML stack), good for data applications, easy deployment, built-in widgets for forms
  - **Why not Flask/FastAPI + React**: Streamlit reduces development time by 80%, built-in state management, automatic reactive updates, focus on functionality over custom UI
- **Design**: Dark/light mode, responsive layout, professional styling
  - **Dark Mode**: Reduces eye strain for data-heavy applications, modern aesthetic, better for dashboard-style interfaces
  - **Professional Styling**: Agricultural stakeholders include researchers, extension agents, agribusiness - need credible, polished interface
- **Charts**: Plotly (interactive gauges, radar plots, bar charts)
  - **Why Plotly**: Interactive charts for data exploration, professional appearance, integrates with Streamlit, agricultural users appreciate data visualization
- **Reports**: ReportLab PDF generation
  - **Why PDF Reports**: Agricultural sector still uses printable reports, archival format, works offline, professional presentation for stakeholders

### **Architecture Benefits**

1. **Modularity**: Each component (ML, RAG, LLM, UI) can be updated independently
2. **Scalability**: FAISS → database, Streamlit → FastAPI, local → cloud deployment paths
3. **Reliability**: Multiple fallbacks ensure system works even with API failures
4. **Transparency**: Source attribution, explainable ML, structured reasoning chain
5. **Deployment Simplicity**: Self-contained, minimal dependencies, works on Streamlit Cloud
6. **Agricultural Focus**: Domain-specific knowledge integration, interpretable models, actionable recommendations

### **Alternative Approaches Considered**

- **Neural Networks**: Rejected due to small dataset size and interpretability requirements
- **OpenAI GPT**: Rejected due to cost concerns and preference for Anthropic's safety focus
- **Elasticsearch**: Rejected due to deployment complexity vs FAISS simplicity
- **React Frontend**: Rejected due to development time vs Streamlit's rapid iteration
- **Microservices**: Rejected due to deployment complexity for academic/demo project

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


| Feature                   | Milestone 1                           | Milestone 2                              |
| ------------------------- | ------------------------------------- | ---------------------------------------- |
| **ML Prediction**         | Yes - Random Forest yield forecasting | Yes - Integrated into agent reasoning    |
| **Data Sources**          | Yes - FAO global crop production      | Yes - + Agronomy guidelines via RAG      |
| **User Interface**        | Yes - Form-based input/output         | Yes - + Conversational chat interface    |
| **Report Generation**     | Yes - PDF reports with charts         | Yes - + Structured agent recommendations |
| **Personalization**       | Yes - Farm-specific predictions       | Yes - + Multi-turn conversation memory   |
| **Knowledge Integration** | No - Static model only                | Yes - Dynamic retrieval + reasoning      |
| **Document Processing**   | No - No file uploads                  | Yes - PDF/TXT upload + analysis          |
| **Deployment**            | Yes - Self-contained Streamlit app    | Yes - + API key configuration            |


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