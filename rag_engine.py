"""
Ask Skillveri — RAG Engine (Vector DB Edition)
===============================================
Retrieval-Augmented Generation with ChromaDB + Sentence Transformers.

Architecture:
  1. Load domain knowledge from JSON
  2. Generate embeddings using sentence-transformers (all-MiniLM-L6-v2)
  3. Store embeddings in ChromaDB (persistent vector database)
  4. Retrieve top-k semantically similar documents for user query
  5. Send retrieved context + query to OpenAI GPT for answer generation
  6. Return structured response with sources

Why ChromaDB + Sentence Transformers?
  - Semantic search: "welding defect" matches "weld flaw" (TF-IDF can't do this)
  - Persistent storage: embeddings are cached, no re-computation on restart
  - Production-ready: ChromaDB scales to millions of documents
  - Free and local: sentence-transformers runs on CPU, no API costs for embeddings
"""

import json
import os
import re
import math
from collections import Counter
from openai import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer


# ─── Vector DB Retriever (ChromaDB + Sentence Transformers) ────────

class VectorRetriever:
    """Semantic search using ChromaDB vector database and sentence-transformers embeddings."""

    def __init__(self, persist_dir=None):
        self.documents = []

        print("Loading embedding model (first time may download ~90MB)...")
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedding model loaded!")

        if persist_dir is None:
            persist_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'chroma_db'
            )

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = None

    def _embed(self, texts):
        """Generate embeddings using sentence-transformers."""
        embeddings = self.embed_model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def build_index(self, documents, force_rebuild=False):
        """Build or load the ChromaDB index from documents."""
        self.documents = documents

        try:
            self.collection = self.client.get_collection(name="skillveri_kb")
            if self.collection.count() == len(documents) and not force_rebuild:
                print(f"Loaded existing vector index: {self.collection.count()} documents")
                return
            else:
                self.client.delete_collection(name="skillveri_kb")
                print("Rebuilding vector index...")
        except Exception:
            print("Creating new vector index...")

        self.collection = self.client.create_collection(
            name="skillveri_kb",
            metadata={"hnsw:space": "cosine"}
        )

        texts = []
        ids = []
        metadatas = []

        for doc in documents:
            text = f"{doc['topic']}. {doc['subtopic']}. {doc['content']}"
            texts.append(text)
            ids.append(doc['id'])
            metadatas.append({
                'skill': doc['skill'],
                'topic': doc['topic'],
                'subtopic': doc['subtopic'],
                'difficulty': doc['difficulty'],
                'related_param': doc.get('related_param', ''),
            })

        embeddings = self._embed(texts)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        print(f"Vector index built: {len(documents)} documents, 384-dim embeddings")

    def retrieve(self, query, top_k=3, skill_filter=None):
        """Retrieve top-k semantically similar documents."""
        if self.collection is None:
            return []

        where_filter = None
        if skill_filter:
            where_filter = {"skill": skill_filter}

        query_embedding = self._embed([query])

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        if results and results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                distance = results['distances'][0][i]
                similarity = 1 - (distance / 2)

                metadata = results['metadatas'][0][i]

                original_doc = next(
                    (d for d in self.documents if d['id'] == doc_id),
                    None
                )

                doc_result = {
                    'id': doc_id,
                    'skill': metadata.get('skill', ''),
                    'topic': metadata.get('topic', ''),
                    'subtopic': metadata.get('subtopic', ''),
                    'difficulty': metadata.get('difficulty', ''),
                    'related_param': metadata.get('related_param', ''),
                    'content': original_doc['content'] if original_doc else results['documents'][0][i],
                    'relevance_score': round(similarity, 4),
                }
                retrieved.append(doc_result)

        return retrieved


# ─── TF-IDF Fallback Retriever ──────────────────────────────────────

def tokenize(text):
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
    """Lightweight TF-IDF fallback if ChromaDB isn't available."""

    def __init__(self):
        self.documents = []
        self.doc_tokens = []
        self.idf = {}
        self.doc_tfidf = []

    def build_index(self, documents):
        self.documents = documents
        self.doc_tokens = []

        for doc in documents:
            text = f"{doc['topic']} {doc['subtopic']} {doc['content']} {doc['skill']}"
            tokens = tokenize(text)
            self.doc_tokens.append(tokens)

        n_docs = len(documents)
        doc_freq = Counter()
        for tokens in self.doc_tokens:
            for token in set(tokens):
                doc_freq[token] += 1

        self.idf = {
            token: math.log((n_docs + 1) / (freq + 1)) + 1
            for token, freq in doc_freq.items()
        }

        self.doc_tfidf = []
        for tokens in self.doc_tokens:
            tf = Counter(tokens)
            total = len(tokens)
            tfidf = {
                token: (count / total) * self.idf.get(token, 1)
                for token, count in tf.items()
            }
            self.doc_tfidf.append(tfidf)

        print(f"TF-IDF index built: {n_docs} documents")

    def retrieve(self, query, top_k=3):
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        query_tf = Counter(query_tokens)
        query_total = len(query_tokens)
        query_tfidf = {
            token: (count / query_total) * self.idf.get(token, 1)
            for token, count in query_tf.items()
        }

        scores = []
        for i, doc_vec in enumerate(self.doc_tfidf):
            dot = sum(
                query_tfidf.get(token, 0) * doc_vec.get(token, 0)
                for token in set(list(query_tfidf.keys()) + list(doc_vec.keys()))
            )
            q_mag = math.sqrt(sum(v ** 2 for v in query_tfidf.values()))
            d_mag = math.sqrt(sum(v ** 2 for v in doc_vec.values()))
            similarity = dot / (q_mag * d_mag) if q_mag > 0 and d_mag > 0 else 0
            scores.append((i, similarity))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scores[:top_k]:
            if score > 0.01:
                doc = self.documents[idx].copy()
                doc['relevance_score'] = round(score, 4)
                results.append(doc)

        return results


# ─── Knowledge Base ─────────────────────────────────────────────────

def load_knowledge_base(kb_path=None):
    if kb_path is None:
        kb_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'domain_knowledge.json'
        )

    with open(kb_path, 'r') as f:
        documents = json.load(f)

    print(f"Loaded {len(documents)} knowledge documents")

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
    """Main RAG pipeline — ChromaDB vector search with TF-IDF fallback."""

    def __init__(self, api_key, model="gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self.documents = load_knowledge_base()
        self.retrieval_method = "vector_db"
        self.conversation_history = []

        try:
            self.retriever = VectorRetriever()
            self.retriever.build_index(self.documents)
            print("Using: ChromaDB + Sentence Transformers (semantic search)")
        except Exception as e:
            print(f"ChromaDB failed ({e}), falling back to TF-IDF...")
            self.retriever = TFIDFRetriever()
            self.retriever.build_index(self.documents)
            self.retrieval_method = "tfidf"
            print("Using: TF-IDF (keyword search)")

    def ask(self, question, top_k=3):
        retrieved = self.retriever.retrieve(question, top_k=top_k)

        try:
            answer, tokens = generate_answer(
                question, retrieved, self.api_key, self.model
            )
            error = None
        except Exception as e:
            answer = f"Sorry, I couldn't generate an answer. Error: {str(e)}"
            tokens = 0
            error = str(e)

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
            'retrieval_method': self.retrieval_method,
            'error': error,
        }

        self.conversation_history.append(result)

        return result

    def get_knowledge_stats(self):
        skills = Counter(doc['skill'] for doc in self.documents)
        difficulties = Counter(doc['difficulty'] for doc in self.documents)

        return {
            'total_articles': len(self.documents),
            'skills_coverage': dict(skills),
            'difficulty_distribution': dict(difficulties),
            'retrieval_method': self.retrieval_method,
        }


if __name__ == '__main__':
    docs = load_knowledge_base()

    test_queries = [
        "Why does my weld have porosity?",
        "How to prevent orange peel in spray painting?",
        "How does refrigeration cycle work?",
        "Solar panel mounting procedure",
        "What is 6G welding position?",
        "weld flaw detection",
        "AC repair cooling problem",
        "coating thickness measurement",
    ]

    print("\n" + "=" * 60)
    print("VECTOR DB TEST (ChromaDB + Sentence Transformers)")
    print("=" * 60)

    try:
        vector_retriever = VectorRetriever()
        vector_retriever.build_index(docs, force_rebuild=True)

        for query in test_queries:
            print(f"\nQuery: {query}")
            results = vector_retriever.retrieve(query, top_k=2)
            for r in results:
                print(f"  -> [{r['relevance_score']:.3f}] {r['skill']}/{r['topic']} — {r['subtopic']}")
    except Exception as e:
        print(f"Vector DB test failed: {e}")

    print("\n" + "=" * 60)
    print("TF-IDF TEST (keyword matching) — for comparison")
    print("=" * 60)

    tfidf_retriever = TFIDFRetriever()
    tfidf_retriever.build_index(docs)

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = tfidf_retriever.retrieve(query, top_k=2)
        for r in results:
            print(f"  -> [{r['relevance_score']:.3f}] {r['skill']}/{r['topic']} — {r['subtopic']}")