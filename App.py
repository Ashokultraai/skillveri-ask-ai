"""
Ask Skillveri — Conversational AI Tutor
=======================================
Streamlit chat interface powered by RAG + OpenAI.
Includes RAGAS evaluation page.

Run: streamlit run App.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))
from src.rag_engine import AskSkillveri, load_knowledge_base, TFIDFRetriever

# ─── Page Config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ask Skillveri — AI Tutor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size: 2rem; font-weight: 700; margin-bottom: 0; }
    .sub-title { font-size: 1.05rem; color: #94a3b8; margin-top: 4px; margin-bottom: 20px; }
    .source-card {
        background: #1e293b !important;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        color: #ffffff !important;
        font-size: 0.85rem;
    }
    .source-card b { color: #ffffff !important; }
    .skill-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .skill-welding { background: #1e3a5f; color: #85B7EB; }
    .skill-spray_painting { background: #2d1a3d; color: #CECBF6; }
    .skill-hvac { background: #0f3d2e; color: #9FE1CB; }
    .skill-solar { background: #3d2a0f; color: #FAC775; }
    .how-it-works {
        background: #1e293b !important;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
        color: #ffffff !important;
    }
    .how-it-works b { color: #ffffff !important; }
    div[data-testid="stSidebar"] { background-color: #111827; }
    .stats-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin: 12px 0;
    }
    .stat-card {
        background: #1e293b;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .stat-value { font-size: 1.5rem; font-weight: 700; color: #00d4aa; }
    .stat-label { font-size: 0.8rem; color: #94a3b8; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ─────────────────────────────────────────────
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'total_tokens' not in st.session_state:
    st.session_state.total_tokens = 0
if 'query_count' not in st.session_state:
    st.session_state.query_count = 0

# ─── Load knowledge base once ───────────────────────────────────────
kb_path = os.path.join(os.path.dirname(__file__), 'knowledge_base', 'domain_knowledge.json')
kb = load_knowledge_base(kb_path)

skills_count = {}
for doc in kb:
    skills_count[doc['skill']] = skills_count.get(doc['skill'], 0) + 1

# ─── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Ask Skillveri")
    st.markdown("*AI Tutor for Vocational Skills*")
    st.markdown("---")

    api_key = os.getenv("OPENAI_API_KEY", "")
    model = "gpt-4o-mini"

    st.markdown("---")

    skill_filter = st.selectbox(
        "Filter by Skill",
        ["All Skills", "Welding", "Spray Painting", "HVAC", "Solar Installation"],
        index=0
    )

    st.markdown("---")
    st.markdown("#### 📚 Knowledge Base")
    st.markdown(f"""<div class="stats-grid">
        <div class="stat-card"><div class="stat-value">{len(kb)}</div><div class="stat-label">Total Articles</div></div>
        <div class="stat-card"><div class="stat-value">4</div><div class="stat-label">Skills Covered</div></div>
    </div>""", unsafe_allow_html=True)

    for skill, count in skills_count.items():
        skill_display = skill.replace('_', ' ').title()
        icon = {'welding': '🔧', 'spray_painting': '🎨', 'hvac': '❄️', 'solar': '☀️'}.get(skill, '📄')
        st.caption(f"{icon} {skill_display}: {count} articles")

    st.markdown("---")
    st.markdown("#### 📊 Session Stats")
    st.caption(f"Questions asked: {st.session_state.query_count}")
    st.caption(f"Tokens used: {st.session_state.total_tokens:,}")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.total_tokens = 0
        st.session_state.query_count = 0
        st.rerun()

    st.markdown("---")
    page = st.radio("Navigate", ["💬 Chat", "📊 RAGAS Evaluation"], index=0)

    st.markdown("---")
    st.caption("**Project 3** — Ask Skillveri AI Tutor")
    st.caption("For Skillveri AI Engineer Interview")


# ═══════════════════════════════════════════════════════════════════
# PAGE 1: CHAT
# ═══════════════════════════════════════════════════════════════════
if page == "💬 Chat":

    st.markdown('<p class="main-title">🎓 Ask Skillveri — AI Tutor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Ask any question about welding, spray painting, HVAC, or solar installation. Powered by RAG + OpenAI.</p>', unsafe_allow_html=True)

    chat_col, info_col = st.columns([3, 1])

    with info_col:
        st.markdown("""<div class="how-it-works">
            <b>How this works:</b><br><br>
            <b>1.</b> You ask a question<br>
            <b>2.</b> AI searches 20+ training articles to find relevant content<br>
            <b>3.</b> OpenAI generates an expert answer using that context<br>
            <b>4.</b> Sources are cited so you can verify<br><br>
            This is called <b>RAG</b> (Retrieval-Augmented Generation) — the AI doesn't hallucinate because it's grounded in Skillveri's actual curriculum.
        </div>""", unsafe_allow_html=True)

        st.markdown("#### 💡 Try asking:")

        skill_map = {
            "All Skills": None,
            "Welding": "welding",
            "Spray Painting": "spray_painting",
            "HVAC": "hvac",
            "Solar Installation": "solar",
        }

        suggestions = {
            'welding': [
                "Why does my weld have porosity?",
                "What's the difference between MIG and TIG?",
                "How do I prevent undercut?",
                "How do I weld vertical (3G)?",
                "What PPE do I need for welding?",
            ],
            'spray_painting': [
                "What is DFT and why does it matter?",
                "Why am I getting orange peel?",
                "What's the right spray gun distance?",
            ],
            'hvac': [
                "How does the refrigeration cycle work?",
                "How do I test for leaks?",
                "How do I braze copper tubing?",
            ],
            'solar': [
                "How do I do a solar site survey?",
                "What's the panel mounting process?",
                "How do I test after installation?",
            ],
        }

        selected_skill = skill_map[skill_filter]
        if selected_skill:
            display_suggestions = suggestions.get(selected_skill, [])
        else:
            display_suggestions = []
            for skill_q in suggestions.values():
                display_suggestions.extend(skill_q[:2])

        for suggestion in display_suggestions:
            if st.button(suggestion, key=f"sug_{suggestion}", use_container_width=True):
                st.session_state.pending_question = suggestion
                st.rerun()

    with chat_col:
        if not api_key:
            st.warning("👆 Enter your OpenAI API key in the sidebar to start chatting. Your key is not stored anywhere.")
            st.markdown("---")

            st.markdown("#### 🔍 Demo Mode — Retrieval Only (no API key needed)")
            st.caption("This shows which knowledge base articles would be retrieved for your question.")

            demo_query = st.text_input("Try a search:", placeholder="e.g., Why does my weld have porosity?")

            if demo_query:
                retriever = TFIDFRetriever()
                retriever.build_index(kb)
                results = retriever.retrieve(demo_query, top_k=3)

                if results:
                    st.markdown(f"**Found {len(results)} relevant articles:**")
                    for r in results:
                        skill_class = f"skill-{r['skill']}"
                        st.markdown(f"""<div class="source-card">
                            <span class="skill-badge {skill_class}">{r['skill'].replace('_',' ')}</span>
                            &nbsp; <b>{r['topic']}</b> — {r['subtopic']}
                            &nbsp; <span style="color: #00d4aa;">({r['relevance_score']:.1%} match)</span>
                            <br><br>{r['content'][:300]}...
                        </div>""", unsafe_allow_html=True)
                else:
                    st.info("No matching articles found. Try a different question.")

            st.stop()

        # ── Chat Interface (API key present) ─────────────────────

        @st.cache_resource
        def get_rag_engine(_api_key, _model):
            return AskSkillveri(api_key=_api_key, model=_model)

        try:
            rag = get_rag_engine(api_key, model)
        except Exception as e:
            st.error(f"Failed to initialize: {e}")
            st.stop()

        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and "sources" in msg:
                    with st.expander(f"📚 Sources ({len(msg['sources'])} articles referenced)"):
                        for src in msg['sources']:
                            skill_class = f"skill-{src['skill']}"
                            st.markdown(f"""<div class="source-card">
                                <span class="skill-badge {skill_class}">{src['skill'].replace('_',' ')}</span>
                                &nbsp; <b>{src['topic']}</b> — {src['subtopic']}
                                &nbsp; <span style="color: #00d4aa;">({src['relevance']:.1%} match)</span>
                            </div>""", unsafe_allow_html=True)

        # Handle pending question from suggested buttons
        pending = st.session_state.pop('pending_question', None)
        user_input = st.chat_input("Ask anything about welding, painting, HVAC, or solar...")
        query = pending or user_input

        if query:
            with st.chat_message("user"):
                st.markdown(query)
            st.session_state.messages.append({"role": "user", "content": query})

            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching knowledge base & generating answer..."):
                    result = rag.ask(query, top_k=3)

                st.markdown(result['answer'])

                if result['sources']:
                    with st.expander(f"📚 Sources ({len(result['sources'])} articles referenced)"):
                        for src in result['sources']:
                            skill_class = f"skill-{src['skill']}"
                            st.markdown(f"""<div class="source-card">
                                <span class="skill-badge {skill_class}">{src['skill'].replace('_',' ')}</span>
                                &nbsp; <b>{src['topic']}</b> — {src['subtopic']}
                                &nbsp; <span style="color: #00d4aa;">({src['relevance']:.1%} match)</span>
                            </div>""", unsafe_allow_html=True)

                if result['tokens_used'] > 0:
                    st.caption(f"Tokens used: {result['tokens_used']:,}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": result['answer'],
                "sources": result['sources'],
            })

            st.session_state.total_tokens += result['tokens_used']
            st.session_state.query_count += 1

            if result['error']:
                st.error(f"API Error: {result['error']}")


# ═══════════════════════════════════════════════════════════════════
# PAGE 2: RAGAS EVALUATION
# ═══════════════════════════════════════════════════════════════════
elif page == "📊 RAGAS Evaluation":

    st.markdown('<p class="main-title">📊 RAGAS Evaluation — How Good Is This RAG?</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Measure retrieval and answer quality with industry-standard metrics</p>', unsafe_allow_html=True)

    eval_mode = st.radio(
        "Evaluation Mode",
        ["✍️ Manual — Paste & Score (instant)", "🤖 Auto — Run Pipeline (uses API)"],
        index=0,
        horizontal=True
    )

    st.markdown("---")

    # ── MANUAL MODE ─────────────────────────────────────────────
    if eval_mode == "✍️ Manual — Paste & Score (instant)":

        st.markdown("### Paste your question, context, and answer to get instant RAGAS scores")
        st.caption("No API calls needed. Scores are computed locally using text analysis.")

        question = st.text_area(
            "Question",
            placeholder="e.g., Why does my weld have porosity?",
            height=68
        )

        ground_truth = st.text_area(
            "Ground Truth / Expected Answer (optional — improves completeness scoring)",
            placeholder="e.g., Porosity is caused by gas pockets trapped in the weld metal...",
            height=100
        )

        context = st.text_area(
            "Retrieved Context (what the RAG retrieved from the knowledge base)",
            placeholder="Paste the retrieved document(s) content here...",
            height=150
        )

        answer = st.text_area(
            "AI Generated Answer (what the LLM responded)",
            placeholder="Paste the AI's answer here...",
            height=150
        )

        if st.button("📊 Score This Answer", type="primary", use_container_width=True):
            if not question or not answer:
                st.warning("Please enter at least a question and an answer.")
            else:
                from src.ragas_eval import (
                    compute_parameter_mention_score,
                    compute_actionability_score,
                    compute_answer_completeness,
                )
                import re

                st.markdown("---")
                st.markdown("### Results")

                # ── Faithfulness (answer grounded in context?) ──
                if context.strip():
                    ctx_words = set(re.findall(r'[a-z0-9]+', context.lower()))
                    ans_words = set(re.findall(r'[a-z0-9]+', answer.lower()))
                    stopwords = {'the','a','an','is','are','was','were','to','of','in','for','on','with','and','but','or','it','its','this','that','be','been','have','has','had','do','does','did','not','at','by','from','as'}
                    ctx_words -= stopwords
                    ans_words -= stopwords
                    if ans_words:
                        faithfulness = min(1.0, len(ans_words & ctx_words) / len(ans_words) * 1.3)
                    else:
                        faithfulness = 0.0
                else:
                    faithfulness = 0.0

                # ── Answer Relevancy (does answer address the question?) ──
                q_words = set(re.findall(r'[a-z0-9]+', question.lower()))
                a_words = set(re.findall(r'[a-z0-9]+', answer.lower()))
                q_words -= {'the','a','an','is','are','to','of','in','for','on','with','and','but','or','what','how','why','does','my','do','i','can','should','would'}
                if q_words:
                    relevancy = min(1.0, len(q_words & a_words) / len(q_words) * 1.5)
                else:
                    relevancy = 0.0

                # ── Context Precision (is context relevant to question?) ──
                if context.strip():
                    c_words = set(re.findall(r'[a-z0-9]+', context.lower()))
                    c_words -= {'the','a','an','is','are','to','of','in','for','on','with','and','but','or'}
                    if q_words:
                        precision = min(1.0, len(q_words & c_words) / len(q_words) * 1.5)
                    else:
                        precision = 0.0
                else:
                    precision = 0.0

                # ── Context Recall (did context have needed info?) ──
                if ground_truth.strip() and context.strip():
                    gt_words = set(re.findall(r'[a-z0-9]+', ground_truth.lower()))
                    gt_words -= {'the','a','an','is','are','to','of','in','for','on','with','and','but','or'}
                    c_words = set(re.findall(r'[a-z0-9]+', context.lower()))
                    if gt_words:
                        recall = min(1.0, len(gt_words & c_words) / len(gt_words) * 1.5)
                    else:
                        recall = 0.0
                else:
                    recall = 0.0

                # ── Custom Skillveri Metrics ──
                param_mention = compute_parameter_mention_score(answer)
                actionability = compute_actionability_score(answer)
                completeness = compute_answer_completeness(answer, ground_truth) if ground_truth.strip() else 0.0

                # ── Display Scores ──
                def metric_color(val):
                    if val >= 0.8: return "🟢"
                    elif val >= 0.6: return "🟡"
                    else: return "🔴"

                st.markdown("#### RAGAS Metrics")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric(f"{metric_color(faithfulness)} Faithfulness", f"{faithfulness:.3f}",
                            help="Is the answer grounded in the retrieved context?")
                col2.metric(f"{metric_color(relevancy)} Answer Relevancy", f"{relevancy:.3f}",
                            help="Does the answer address the question?")
                col3.metric(f"{metric_color(precision)} Context Precision", f"{precision:.3f}",
                            help="Is the retrieved context relevant to the question?")
                col4.metric(f"{metric_color(recall)} Context Recall", f"{recall:.3f}",
                            help="Did the context contain the needed information?")

                if not context.strip():
                    st.caption("⚠️ No context provided — Faithfulness, Precision, and Recall may be inaccurate.")
                if not ground_truth.strip():
                    st.caption("⚠️ No ground truth provided — Recall and Completeness scored as 0.")

                st.markdown("#### Skillveri Custom Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric(f"{metric_color(param_mention)} Parameter Mention", f"{param_mention:.3f}",
                            help="Does answer reference simulator parameters?")
                col2.metric(f"{metric_color(actionability)} Actionability", f"{actionability:.3f}",
                            help="Does answer give specific, practical advice?")
                col3.metric(f"{metric_color(completeness)} Completeness", f"{completeness:.3f}",
                            help="How much of the ground truth is covered?")

                # Overall score
                avg_score = (faithfulness + relevancy + precision + param_mention + actionability) / 5
                if ground_truth.strip():
                    avg_score = (faithfulness + relevancy + precision + recall + param_mention + actionability + completeness) / 7

                st.markdown("---")
                overall_emoji = "✅ Excellent" if avg_score >= 0.75 else "🟡 Decent" if avg_score >= 0.5 else "🔴 Needs Improvement"
                st.markdown(f"### Overall Score: {avg_score:.3f} — {overall_emoji}")

    # ── AUTO MODE ───────────────────────────────────────────────
    else:

        st.markdown("""<div class="how-it-works">
            <b>Auto mode</b> runs pre-built test questions through the full RAG pipeline (retrieve → generate → score). 
            Requires OpenAI API calls for answer generation.
        </div>""", unsafe_allow_html=True)

        eval_dir = os.path.join(os.path.dirname(__file__), 'evaluation')
        report_path = os.path.join(eval_dir, 'ragas_report.json')

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            num_q = st.selectbox("Questions to test", [1, 3, 5, 12], index=1)
        with col2:
            if api_key:
                run_eval = st.button("🚀 Run Evaluation", type="primary", use_container_width=True)
            else:
                st.warning("No API key found in .env")
                run_eval = False
        with col3:
            st.caption(f"Runs {num_q} test questions. Takes ~{num_q * 10} seconds.")

        if run_eval and api_key:
            from src.ragas_eval import run_evaluation

            with st.spinner(f"Evaluating {num_q} questions... ~{num_q * 10} seconds"):
                report = run_evaluation(api_key, model, num_questions=num_q, verbose=False)

            st.success("Evaluation complete!")
            st.session_state.eval_report = report

        report = st.session_state.get('eval_report', None)
        if report is None and os.path.exists(report_path):
            with open(report_path) as f:
                report = json.load(f)

        if report is None:
            st.info("No evaluation results yet. Click 'Run Evaluation' above.")
            st.stop()

        # ── Display Results ──
        st.markdown("---")
        st.markdown("### RAGAS Metrics (Industry Standard)")

        ragas = report['ragas_metrics']

        def metric_color(val):
            if val >= 0.8:
                return "🟢"
            elif val >= 0.6:
                return "🟡"
            else:
                return "🔴"

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            f"{metric_color(ragas['faithfulness'])} Faithfulness",
            f"{ragas['faithfulness']:.3f}",
            help="Is the answer grounded in context? 1.0 = no hallucination"
        )
        col2.metric(
            f"{metric_color(ragas['answer_relevancy'])} Answer Relevancy",
            f"{ragas['answer_relevancy']:.3f}",
            help="Does the answer address the question? 1.0 = perfectly on-topic"
        )
        col3.metric(
            f"{metric_color(ragas['context_precision'])} Context Precision",
            f"{ragas['context_precision']:.3f}",
            help="Are retrieved docs relevant? 1.0 = all top docs are relevant"
        )
        col4.metric(
            f"{metric_color(ragas['context_recall'])} Context Recall",
            f"{ragas['context_recall']:.3f}",
            help="Did retrieval find all needed info? 1.0 = nothing missed"
        )

        st.caption(f"Method: {ragas.get('method', 'unknown')}")

        # Custom metrics
        st.markdown("### Skillveri Custom Metrics")

        custom = report['custom_metrics']
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            f"{metric_color(custom['context_hit_rate'])} Context Hit Rate",
            f"{custom['context_hit_rate']:.3f}",
            help="Did retrieval find the right article for each question?"
        )
        col2.metric(
            f"{metric_color(custom['parameter_mention'])} Param Mention",
            f"{custom['parameter_mention']:.3f}",
            help="Does answer reference Skillveri simulator parameters?"
        )
        col3.metric(
            f"{metric_color(custom['actionability'])} Actionability",
            f"{custom['actionability']:.3f}",
            help="Does answer give specific, practical advice with numbers?"
        )
        col4.metric(
            f"{metric_color(custom['answer_completeness'])} Completeness",
            f"{custom['answer_completeness']:.3f}",
            help="How much of the expected answer info is covered?"
        )

        # Per-question breakdown
        st.markdown("---")
        st.markdown("### Per-Question Results")
        st.caption("Expand each question to see the AI's answer, sources retrieved, and individual scores.")

        for i, result in enumerate(report['per_question_results']):
            score_avg = (result['context_hit_rate'] + result['actionability']) / 2
            score_emoji = "✅" if score_avg >= 0.5 else "⚠️"

            with st.expander(f"{score_emoji} Q{i+1}: {result['question']}", expanded=False):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown("**AI Answer:**")
                    st.markdown(result['answer'])

                    st.markdown("**Expected Answer (Ground Truth):**")
                    gt = result['ground_truth']
                    st.caption(gt[:300] + "..." if len(gt) > 300 else gt)

                    st.markdown("**Sources Retrieved:**")
                    for src in result.get('sources_retrieved', []):
                        st.caption(f"  → [{src['relevance']:.1%}] {src['skill']}/{src['topic']}")

                with col2:
                    st.markdown("**Scores:**")
                    st.metric("Context Hit", f"{result['context_hit_rate']:.2f}")
                    st.metric("Param Mention", f"{result['parameter_mention']:.2f}")
                    st.metric("Actionability", f"{result['actionability']:.2f}")
                    st.metric("Completeness", f"{result['answer_completeness']:.2f}")

        # Summary
        st.markdown("---")
        st.markdown(
            f"**Total tokens used:** {report.get('total_tokens', 0):,} | "
            f"**Model:** {report.get('model', 'unknown')} | "
            f"**Questions tested:** {report.get('num_questions', 0)}"
        )