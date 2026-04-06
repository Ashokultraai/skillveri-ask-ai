# 🎓 Ask Skillveri — RAG AI Tutor with Vector DB

**An LLM-powered chatbot that answers welding, painting, HVAC, and solar questions — grounded in Skillveri's curriculum using RAG with ChromaDB semantic search.**

Built for the **Skillveri AI Engineer** interview — Project 3 of 3.

---

## The Problem

Students using Skillveri simulators constantly have theory questions: "Why does my weld have porosity?" "What's the right DFT?" Currently they need an instructor — who isn't always available, especially for 60+ US CTE schools using Skillveri remotely.

## The Solution

**Ask Skillveri** — a RAG-powered AI tutor that:
1. Searches a curated knowledge base using **ChromaDB vector database + sentence-transformers** for semantic search
2. Retrieves the most relevant content (understands meaning, not just keywords)
3. Sends the context + question to **OpenAI GPT-4o-mini** for expert-level answers
4. Cites sources so answers are verifiable, not hallucinated
5. Evaluates answer quality using **RAGAS metrics**

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Vector Database | **ChromaDB** | Persistent, local, scales to millions of docs |
| Embeddings | **sentence-transformers (all-MiniLM-L6-v2)** | 384-dim embeddings, runs on CPU, free |
| LLM | **OpenAI GPT-4o-mini** | Fast, cheap ($0.15/1M tokens), high quality |
| Evaluation | **RAGAS** | Industry standard RAG evaluation framework |
| UI | **Streamlit** | Interactive chat + evaluation dashboard |
| Fallback | **TF-IDF** | Auto-fallback if ChromaDB fails to load |

### Why ChromaDB + Sentence Transformers over TF-IDF?

| Query | TF-IDF finds | Vector DB finds |
|-------|-------------|----------------|
| "weld flaw detection" | Nothing (no exact word match) | Porosity, Undercut, Lack of Fusion |
| "AC repair cooling problem" | Nothing | Refrigeration Cycle |
| "coating thickness measurement" | Nothing | DFT (Dry Film Thickness) |

**Semantic search understands meaning**, not just keywords. "Weld flaw" matches "porosity" because they mean similar things, even though they share zero words.

---

## Project Structure

```
skillveri-ask-ai/
├── App.py                    # Streamlit UI: Chat + RAGAS Evaluation
├── rag_engine.py             # Core RAG: ChromaDB retriever + OpenAI generation
├── ragas_eval.py             # RAGAS evaluation with 12 test questions
├── domain_knowledge.json     # 23 curated articles (the knowledge base)
├── requirements.txt          # Dependencies
├── README.md                 # This file
├── .env                      # Your OpenAI API key (DO NOT upload to GitHub)
├── chroma_db/                # Auto-created: ChromaDB persistent storage
└── evaluation/               # Auto-created: RAGAS evaluation reports
```

## Quick Start

```bash
pip install -r requirements.txt
streamlit run App.py
```

Make sure your `.env` file has: `OPENAI_API_KEY=sk-your-key-here`

First run takes ~30 seconds to download the embedding model (~90MB). After that, the vector index is cached and loads instantly.

---

## How It Works

```
User: "Why does my weld have porosity?"
                    |
                    v
    +-------------------------------+
    |  Sentence Transformer         |
    |  Converts question to         |
    |  384-dim embedding vector     |
    +---------------+---------------+
                    |
                    v
    +-------------------------------+
    |  ChromaDB Vector Search       |
    |  Finds top 3 most similar     |
    |  articles by cosine distance  |
    |  (semantic, not keyword)      |
    +---------------+---------------+
                    |  3 relevant articles
                    v
    +-------------------------------+
    |  OpenAI GPT-4o-mini           |
    |  System: "You are Ask         |
    |  Skillveri, an expert tutor"  |
    |  Context: [3 articles]        |
    |  Question: user's query       |
    +---------------+---------------+
                    |
                    v
    +-------------------------------+
    |  Answer + Source Citations     |
    |  Displayed in Streamlit chat  |
    +-------------------------------+
```

## How Files Connect

```
domain_knowledge.json    (23 articles — the brain)
        |
        v
rag_engine.py            (loads articles, builds ChromaDB index,
        |                 handles retrieval + OpenAI generation)
        |
        +-------> App.py  (Chat UI — calls rag_engine.ask())
        |
        +-------> ragas_eval.py  (Evaluation — tests 12 questions,
                                  computes RAGAS + custom metrics)
```

## Knowledge Base Coverage

| Skill | Articles | Key Topics |
|-------|----------|------------|
| Welding | 12 | Porosity, undercut, LOF, GMAW/GTAW/SMAW, positions 1G-6G, safety, certification |
| Spray Painting | 4 | DFT, gun technique, defects (orange peel, runs, fish eyes), spray processes |
| HVAC | 4 | Refrigeration cycle, leak testing, brazing, refrigerant charging |
| Solar | 3 | Site survey, panel installation, electrical testing |

## RAGAS Evaluation Metrics

| Metric | What It Measures | Good Score |
|--------|-----------------|------------|
| Faithfulness | Answer grounded in context? No hallucination? | >= 0.80 |
| Answer Relevancy | Does answer address the question? | >= 0.80 |
| Context Precision | Were retrieved docs relevant? | >= 0.70 |
| Context Recall | Did retrieval find all needed info? | >= 0.70 |
| Parameter Mention | Does answer reference simulator params? | >= 0.40 |
| Actionability | Specific numbers, steps, practical fixes? | >= 0.60 |

---

## Business Impact

- **Replaces instructor dependency** — students get expert answers 24/7
- **Semantic search** — understands meaning, not just keywords (competitive advantage)
- **No hallucination** — RAG grounds answers in verified curriculum
- **Embeddable in VR** — could run as voice assistant inside Meta Quest headsets
- **vs Interplay's SAM** — deeper, curriculum-specific, with RAGAS quality proof

---

*Built for the Skillveri AI Engineer interview, April 2026*