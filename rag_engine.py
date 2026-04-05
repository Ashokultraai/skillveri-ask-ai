"""
Ask Skillveri — RAG Engine
==========================
Retrieval-Augmented Generation for Skillveri domain knowledge.

Architecture:
  1. Load domain knowledge from JSON
  2. Build TF-IDF index for fast retrieval (no heavy dependencies)
  3. Retrieve top-k relevant documents for user query
  4. Send retrieved context + query to OpenAI GPT for answer generation
  5. Return structured response with sources

Why TF-IDF instead of embeddings?
  - Zero external dependencies (no ChromaDB, no sentence-transformers)
  - Fast to build and query (< 1ms)
  - Works offline for the retrieval step
  - For a 20-document knowledge base, TF-IDF performs comparably to embeddings
  - In production, would upgrade to OpenAI embeddings + Pinecone/Weaviate
"""

import json
import os
import re
import math
from collections import Counter
from openai import OpenAI


# ─── TF-IDF Retrieval Engine ───────────────────────────────────────

def tokenize(text):
    """Simple tokenizer: lowercase, split on non-alphanumeric, remove stopwords."""
    stopwords = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
        'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'and', 'but', 'or',
        'nor', 'not', 'so', 'yet', 'both', 'either', 'neither', 'each',
        'every', 'all', 'any', 'few', 'more', 'most', 'other', 'some',
        'such', 'no', 'only', 'own', 'same', 'than', 'too', 'very',
        'it', 'its', 'this', 'that', 'these', 'those', 'i', 'me', 'my',
        'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
        'they', 'them', 'their', 'what', 'which', 'who', 'whom',
    }
    tokens = re.findall(r'[a-z0-9]+', text.lower())
    return [t for t in tokens if t not in stopwords and len(t) > 1]


class TFIDFRetriever:
    """Lightweight TF-IDF based document retriever."""
    
    def __init__(self):
        self.documents = []
        self.doc_tokens = []
        self.idf = {}
        self.doc_tfidf = []
    
    def build_index(self, documents):
        """Build TF-IDF index from a list of document dicts."""
        self.documents = documents
        self.doc_tokens = []
        
        # Tokenize all documents
        for doc in documents:
            # Combine all searchable fields
            text = f"{doc['topic']} {doc['subtopic']} {doc['content']} {doc['skill']}"
            tokens = tokenize(text)
            self.doc_tokens.append(tokens)
        
        # Compute IDF
        n_docs = len(documents)
        all_tokens = set()
        for tokens in self.doc_tokens:
            all_tokens.update(set(tokens))
        
        doc_freq = Counter()
        for tokens in self.doc_tokens:
            for token in set(tokens):
                doc_freq[token] += 1
        
        self.idf = {
            token: math.log((n_docs + 1) / (freq + 1)) + 1
            for token, freq in doc_freq.items()
        }
        
        # Pre-compute TF-IDF vectors for all documents
        self.doc_tfidf = []
        for tokens in self.doc_tokens:
            tf = Counter(tokens)
            total = len(tokens)
            tfidf = {
                token: (count / total) * self.idf.get(token, 1)
                for token, count in tf.items()
            }
            self.doc_tfidf.append(tfidf)
        
        print(f"Index built: {n_docs} documents, {len(all_tokens)} unique tokens")
    
    def retrieve(self, query, top_k=3):
        """Retrieve top-k most relevant documents for a query."""
        query_tokens = tokenize(query)
        
        if not query_tokens:
            return []
        
        # Compute query TF-IDF
        query_tf = Counter(query_tokens)
        query_total = len(query_tokens)
        query_tfidf = {
            token: (count / query_total) * self.idf.get(token, 1)
            for token, count in query_tf.items()
        }
        
        # Compute cosine similarity with each document
        scores = []
        for i, doc_vec in enumerate(self.doc_tfidf):
            # Dot product
            dot = sum(
                query_tfidf.get(token, 0) * doc_vec.get(token, 0)
                for token in set(list(query_tfidf.keys()) + list(doc_vec.keys()))
            )
            
            # Magnitudes
            q_mag = math.sqrt(sum(v ** 2 for v in query_tfidf.values()))
            d_mag = math.sqrt(sum(v ** 2 for v in doc_vec.values()))
            
            if q_mag > 0 and d_mag > 0:
                similarity = dot / (q_mag * d_mag)
            else:
                similarity = 0
            
            scores.append((i, similarity))
        
        # Sort by similarity (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k with similarity > 0
        results = []
        for idx, score in scores[:top_k]:
            if score > 0.01:
                doc = self.documents[idx].copy()
                doc['relevance_score'] = round(score, 4)
                results.append(doc)
        
        return results


# ─── Knowledge Base ─────────────────────────────────────────────────

def load_knowledge_base(kb_path=None):
    """Load the domain knowledge JSON file."""
    if kb_path is None:
        kb_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'knowledge_base', 'domain_knowledge.json'
        )
    
    with open(kb_path, 'r') as f:
        documents = json.load(f)
    
    print(f"Loaded {len(documents)} knowledge documents")
    
    # Print coverage summary
    skills = Counter(doc['skill'] for doc in documents)
    for skill, count in skills.items():
        print(f"  {skill}: {count} articles")
    
    return documents


# ─── OpenAI LLM Integration ────────────────────────────────────────

SYSTEM_PROMPT = """You are "Ask Skillveri" — an expert AI tutor for vocational skill training. You help students and trainers understand welding, spray painting, HVAC/refrigeration, and solar panel installation.

Your knowledge comes from Skillveri's training materials. You are integrated with Skillveri's VR/MR simulators (AURA for welding, Chroma for painting, Kelvin for HVAC, Solis for solar).

RULES:
1. Answer questions clearly and practically — these are vocational learners, not academics.
2. Always relate your answer back to the Skillveri simulator parameters when relevant (work_angle, travel_angle, travel_speed, contact_tip_distance, arc_length, bead_quality).
3. Use the CONTEXT provided from the knowledge base. If the context answers the question, use it. If not, use your general knowledge but note that.
4. Give specific, actionable advice — not vague general statements.
5. When mentioning numbers (temperatures, pressures, thicknesses), include both metric and imperial units where applicable.
6. Use simple language. Avoid jargon unless you immediately define it.
7. If the question is about a topic you don't have information on, say so honestly and suggest what the student should ask their instructor.
8. Structure longer answers with clear sections but keep it conversational.
9. If the student describes a problem they're having in the simulator, diagnose the likely cause and suggest specific fixes.

SIMULATOR PARAMETERS (score 0-100 each):
- work_angle: torch angle relative to workpiece (deviation from ideal)
- travel_angle: forward/backward tilt of torch (push vs drag)  
- travel_speed: consistency of movement along the joint
- contact_tip_distance (CTWD): distance from gun tip to workpiece
- arc_length: stability of the electrical arc
- bead_quality: overall appearance of the finished weld bead"""


def generate_answer(query, retrieved_docs, api_key, model="gpt-4o-mini"):
    """Generate an answer using OpenAI API with retrieved context."""
    
    # Format retrieved context
    context_parts = []
    for i, doc in enumerate(retrieved_docs):
        context_parts.append(
            f"[Source {i+1}: {doc['skill'].upper()} — {doc['topic']} > {doc['subtopic']}]\n"
            f"{doc['content']}"
        )
    
    context_text = "\n\n---\n\n".join(context_parts)
    
    user_message = f"""CONTEXT FROM KNOWLEDGE BASE:
{context_text}

---

STUDENT QUESTION: {query}

Please answer the student's question using the context above. If the context is relevant, base your answer on it and mention specific Skillveri simulator features. If the context doesn't fully cover the question, supplement with your general knowledge but prioritize the provided context."""

    client = OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=0.3,
        max_tokens=1000,
    )
    
    answer = response.choices[0].message.content
    tokens_used = response.usage.total_tokens if response.usage else 0
    
    return answer, tokens_used


# ─── Main RAG Pipeline ─────────────────────────────────────────────

class AskSkillveri:
    """Main RAG pipeline — ties retrieval and generation together."""
    
    def __init__(self, api_key, model="gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self.retriever = TFIDFRetriever()
        self.documents = load_knowledge_base()
        self.retriever.build_index(self.documents)
        self.conversation_history = []
    
    def ask(self, question, top_k=3):
        """Full RAG pipeline: retrieve → generate → return."""
        
        # Step 1: Retrieve relevant documents
        retrieved = self.retriever.retrieve(question, top_k=top_k)
        
        # Step 2: Generate answer with LLM
        try:
            answer, tokens = generate_answer(
                question, retrieved, self.api_key, self.model
            )
            error = None
        except Exception as e:
            answer = f"Sorry, I couldn't generate an answer. Error: {str(e)}"
            tokens = 0
            error = str(e)
        
        # Step 3: Build response
        result = {
            'question': question,
            'answer': answer,
            'sources': [
                {
                    'id': doc['id'],
                    'skill': doc['skill'],
                    'topic': doc['topic'],
                    'subtopic': doc['subtopic'],
                    'relevance': doc['relevance_score'],
                    'difficulty': doc['difficulty'],
                }
                for doc in retrieved
            ],
            'tokens_used': tokens,
            'error': error,
        }
        
        # Store in conversation history
        self.conversation_history.append(result)
        
        return result
    
    def get_suggested_questions(self, skill=None):
        """Return suggested questions based on the knowledge base."""
        suggestions = {
            'welding': [
                "Why does my weld have porosity?",
                "What's the difference between MIG and TIG welding?",
                "How do I prevent undercut?",
                "What electrode should I use for structural steel?",
                "Why is my travel speed score low in the simulator?",
                "How do I weld in the vertical (3G) position?",
                "What does lack of fusion mean and how do I fix it?",
                "What safety equipment do I need for welding?",
                "How do I prepare for AWS certification?",
            ],
            'spray_painting': [
                "What is DFT and why does it matter?",
                "How far should I hold the spray gun from the surface?",
                "Why am I getting orange peel texture?",
                "What's the difference between HVLP and airless spraying?",
                "How do I prevent runs and sags?",
                "What causes fish eyes in paint?",
            ],
            'hvac': [
                "How does the refrigeration cycle work?",
                "How do I test for refrigerant leaks?",
                "What is superheat and subcooling?",
                "How do I braze copper tubing properly?",
                "How do I charge a system with R-410A?",
                "What is the EPA 608 certification?",
            ],
            'solar': [
                "How do I conduct a solar site survey?",
                "What's the correct panel mounting procedure?",
                "How do I test a solar system after installation?",
                "What affects solar panel orientation?",
                "How do I check for shading issues?",
            ],
        }
        
        if skill and skill in suggestions:
            return suggestions[skill]
        
        # Return a mix
        all_suggestions = []
        for skill_questions in suggestions.values():
            all_suggestions.extend(skill_questions[:3])
        return all_suggestions
    
    def get_knowledge_stats(self):
        """Return statistics about the knowledge base."""
        skills = Counter(doc['skill'] for doc in self.documents)
        difficulties = Counter(doc['difficulty'] for doc in self.documents)
        topics = [doc['topic'] for doc in self.documents]
        
        return {
            'total_articles': len(self.documents),
            'skills_coverage': dict(skills),
            'difficulty_distribution': dict(difficulties),
            'topics': topics,
        }


if __name__ == '__main__':
    # Test the retrieval engine (no API key needed)
    docs = load_knowledge_base()
    retriever = TFIDFRetriever()
    retriever.build_index(docs)
    
    test_queries = [
        "Why does my weld have porosity?",
        "How to prevent orange peel in spray painting?",
        "How does refrigeration cycle work?",
        "Solar panel mounting procedure",
        "What is 6G welding position?",
    ]
    
    print("\n" + "=" * 60)
    print("RETRIEVAL TEST (no LLM)")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retriever.retrieve(query, top_k=2)
        for r in results:
            print(f"  → [{r['relevance_score']:.3f}] {r['skill']}/{r['topic']} — {r['subtopic']}")