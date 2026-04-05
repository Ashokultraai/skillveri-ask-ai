"""
Ask Skillveri — RAGAS Evaluation Module
========================================
Evaluates the RAG pipeline using RAGAS metrics:

1. Faithfulness    — Is the answer grounded in the retrieved context? (no hallucination)
2. Answer Relevancy — Does the answer actually address the question?
3. Context Precision — Are the retrieved docs relevant to the question?
4. Context Recall   — Did retrieval capture all needed information?

Also includes custom domain-specific metrics:
5. Skillveri Parameter Mention — Does the answer reference simulator parameters?
6. Actionability Score — Does the answer give specific, practical advice?

Run: python src/ragas_eval.py (requires OPENAI_API_KEY env variable or pass it)
"""

import json
import os
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.rag_engine import AskSkillveri, TFIDFRetriever, load_knowledge_base, generate_answer


# ─── Test Dataset (Golden QA pairs for evaluation) ──────────────────

EVAL_DATASET = [
    {
        "question": "Why does my weld have porosity?",
        "ground_truth": "Porosity is caused by gas pockets trapped in the weld metal. Common causes include contaminated base metal (oil, grease, rust, moisture), insufficient shielding gas flow (below 15 CFH for MIG), excessive travel speed not allowing gas to escape, long arc length allowing atmospheric contamination, wind or drafts blowing away shielding gas, and moisture in electrode or flux. Prevention includes cleaning the joint area, checking gas flow rate (25-35 CFH for GMAW), maintaining proper CTWD of 10-15mm, and using wind screens outdoors.",
        "expected_skill": "welding",
        "expected_topic": "Porosity",
        "difficulty": "beginner"
    },
    {
        "question": "What is the difference between MIG and TIG welding?",
        "ground_truth": "GMAW (MIG) uses a continuous wire electrode fed through a welding gun with shielding gas. It's faster and easier to learn. GTAW (TIG) uses a non-consumable tungsten electrode with a separate filler rod added manually. TIG produces higher quality welds but requires more skill. TIG uses 100% Argon gas, push technique only, and requires the shortest arc length. MIG uses 75/25 Ar/CO2 mix and can use push or drag technique. TIG is preferred for precision work, MIG for production speed.",
        "expected_skill": "welding",
        "expected_topic": "GMAW",
        "difficulty": "beginner"
    },
    {
        "question": "How do I prevent undercut in welding?",
        "ground_truth": "Undercut is a groove melted into the base metal along the toe of the weld. Prevent it by: reducing current/amperage by 5-10A, maintaining correct work angle (45° for fillet welds, 90° for butt joints), reducing travel speed to allow filler metal to fill the groove, and keeping weave width within 2-3 times the electrode diameter. In Skillveri's AURA simulator, undercut is detected through the work_angle and travel_speed parameters.",
        "expected_skill": "welding",
        "expected_topic": "Undercut",
        "difficulty": "beginner"
    },
    {
        "question": "What is DFT in spray painting and why does it matter?",
        "ground_truth": "Dry Film Thickness (DFT) is the thickness of the paint coating after drying, measured in microns or mils. Typical values: automotive primer 25-40 microns, basecoat 12-25 microns, clearcoat 40-75 microns. Too thin causes inadequate coverage and poor corrosion protection. Too thick causes sagging, orange peel, cracking, and wasted material. Skillveri's Chroma simulator provides a real-time DFT thickness map showing distribution across the surface.",
        "expected_skill": "spray_painting",
        "expected_topic": "Dry Film Thickness",
        "difficulty": "beginner"
    },
    {
        "question": "How does the refrigeration cycle work?",
        "ground_truth": "The refrigeration cycle has four stages: (1) Compressor compresses low-pressure gas into high-pressure, high-temperature gas. (2) Condenser (outdoor coil) releases heat and condenses gas into high-pressure liquid. (3) Expansion device (TXV) reduces pressure, creating cold low-pressure liquid/gas mix. (4) Evaporator (indoor coil) absorbs heat from indoor air, evaporating refrigerant into low-pressure gas. Key measurements: superheat (8-12°F) and subcooling (10-15°F) indicate proper system charge.",
        "expected_skill": "hvac",
        "expected_topic": "Refrigeration Cycle",
        "difficulty": "beginner"
    },
    {
        "question": "How do I test for refrigerant leaks?",
        "ground_truth": "Leak testing methods: (1) Nitrogen pressure test — charge to 150-300 PSI, monitor for 24 hours. (2) Electronic leak detector — handheld sniffer moved along joints and connections. (3) Soap bubble test — apply soap solution and look for bubbles. (4) UV dye test — inject fluorescent dye, use UV light to find leaks. (5) Standing vacuum test — pull to 500 microns, monitor for rise. Common leak locations: Schrader valves, flare fittings, brazed joints, compressor shaft seals.",
        "expected_skill": "hvac",
        "expected_topic": "Leak Testing",
        "difficulty": "beginner"
    },
    {
        "question": "What is the correct spray gun distance for HVLP?",
        "ground_truth": "For HVLP (High Volume Low Pressure) spray painting, maintain 10-12 inches (25-30 cm) gun distance from the surface. Too close causes heavy buildup, runs, sags, and orange peel. Too far causes dry spray, poor coverage, overspray waste, and rough sandy texture. Travel speed should be 12-16 inches per second with 50% overlap between passes. Keep the gun perpendicular to the surface — no arcing or fanning.",
        "expected_skill": "spray_painting",
        "expected_topic": "Spray Gun Technique",
        "difficulty": "beginner"
    },
    {
        "question": "How do I do a solar site survey?",
        "ground_truth": "Site survey assessment factors: (1) Roof orientation — south-facing is best in Northern Hemisphere. (2) Roof tilt — optimal angle equals latitude. (3) Shading analysis — use Solar Pathfinder to check for shade from trees, buildings, chimneys throughout the year. (4) Roof condition — must have 20+ years remaining life. (5) Electrical assessment — evaluate panel capacity, available breaker spaces, and wire run distance. Skillveri's Solis simulator covers the complete site survey process.",
        "expected_skill": "solar",
        "expected_topic": "Site Survey",
        "difficulty": "beginner"
    },
    {
        "question": "What does 6G welding position mean?",
        "ground_truth": "6G is the most difficult welding position — a pipe fixed at a 45-degree angle that cannot be rotated. The welder must weld all around the pipe, transitioning through flat, vertical, and overhead positions within a single weld. Passing a 6G certification test qualifies the welder for all positions within that process and material. In Skillveri's simulator, 6G has a difficulty multiplier of 1.9x compared to 1.0x for flat (1G) position.",
        "expected_skill": "welding",
        "expected_topic": "Welding Positions",
        "difficulty": "beginner"
    },
    {
        "question": "Why am I getting orange peel texture when painting?",
        "ground_truth": "Orange peel is a textured surface resembling an orange skin. Causes include: gun held too far from surface, paint too thick or not enough reducer added, spray pressure too low, and ambient temperature too hot causing paint to dry before leveling. Fix by adjusting gun distance closer, adding reducer to thin the paint, increasing spray pressure, and controlling temperature. The correct HVLP gun distance is 10-12 inches.",
        "expected_skill": "spray_painting",
        "expected_topic": "Common Spray Painting Defects",
        "difficulty": "intermediate"
    },
    {
        "question": "How do I braze copper tubing for HVAC?",
        "ground_truth": "Brazing procedure: (1) Cut tube square with a tube cutter, deburr internally. (2) Clean tube end and fitting socket with emery cloth until shiny. (3) Apply flux for copper-to-brass joints. (4) Assemble the joint. (5) Flow nitrogen at 2-5 CFH during brazing to prevent internal oxidation. (6) Heat the heavier member (fitting) more than the tube with oxy-acetylene torch. (7) Apply BCuP-6 or Silfos-15 brazing alloy — it should flow by capillary action, not be melted directly by flame. (8) Allow natural cooling, do not quench.",
        "expected_skill": "hvac",
        "expected_topic": "Brazing for HVAC",
        "difficulty": "intermediate"
    },
    {
        "question": "What safety equipment do I need for welding?",
        "ground_truth": "Welding PPE requirements: Welding helmet with correct shade lens (GMAW shade 10-13, GTAW shade 8-13, SMAW shade 10-14). Flame-resistant long-sleeve clothing (leather or FR cotton, no synthetics). Welding gloves with longer cuffs for overhead work. Leather boots. Fume extraction or ventilation system. Fire extinguisher within 10 feet. Clear flammable materials within 35 feet. Inspect cables for damage before use. Never weld in wet conditions.",
        "expected_skill": "welding",
        "expected_topic": "Safety in Welding",
        "difficulty": "beginner"
    },
]


# ─── Custom Metrics (no LLM needed) ─────────────────────────────────

SKILLVERI_PARAMS = [
    'work_angle', 'travel_angle', 'travel_speed',
    'contact_tip_distance', 'ctwd', 'arc_length', 'bead_quality',
    'work angle', 'travel angle', 'travel speed', 'arc length', 'bead quality',
]

def compute_parameter_mention_score(answer):
    """Does the answer reference Skillveri simulator parameters?"""
    answer_lower = answer.lower()
    mentioned = [p for p in SKILLVERI_PARAMS if p in answer_lower]
    # Also check for 'skillveri', 'simulator', 'aura', 'chroma', 'kelvin', 'solis'
    brand_mentions = [b for b in ['skillveri', 'simulator', 'aura', 'chroma', 'kelvin', 'solis']
                      if b in answer_lower]
    
    param_score = min(1.0, len(set(mentioned)) / 2)  # At least 2 params = perfect
    brand_score = min(1.0, len(brand_mentions) / 1)    # At least 1 brand mention = perfect
    
    return round((param_score * 0.6 + brand_score * 0.4), 4)


def compute_actionability_score(answer):
    """Does the answer give specific, practical advice?"""
    answer_lower = answer.lower()
    
    actionable_indicators = [
        # Specific numbers
        any(c.isdigit() for c in answer),
        # Action verbs
        any(word in answer_lower for word in [
            'reduce', 'increase', 'adjust', 'maintain', 'check', 'clean',
            'use', 'apply', 'set', 'hold', 'keep', 'move', 'slow down',
            'speed up', 'practice', 'ensure', 'verify', 'inspect'
        ]),
        # Step-by-step indicators
        any(indicator in answer_lower for indicator in [
            'step', '(1)', '(2)', '1.', '2.', 'first', 'then', 'next', 'finally'
        ]),
        # Specific measurements
        any(unit in answer_lower for unit in [
            'mm', 'inch', 'psi', 'cfh', 'micron', 'degree', '°', 'amp', 'volt'
        ]),
        # Cause-fix structure
        any(word in answer_lower for word in ['cause', 'fix', 'prevent', 'solution', 'because']),
    ]
    
    score = sum(actionable_indicators) / len(actionable_indicators)
    return round(score, 4)


def compute_context_hit_rate(retrieved_docs, expected_skill, expected_topic):
    """Did retrieval find the right documents?"""
    if not retrieved_docs:
        return 0.0
    
    # Check if the expected topic is in the top-k results
    topic_hit = any(
        expected_topic.lower() in doc.get('topic', '').lower()
        for doc in retrieved_docs
    )
    
    # Check if the expected skill is in the top-k results
    skill_hit = any(
        doc.get('skill', '') == expected_skill
        for doc in retrieved_docs
    )
    
    # Check if the top-1 result is the right one
    top1_hit = (
        expected_topic.lower() in retrieved_docs[0].get('topic', '').lower()
        if retrieved_docs else False
    )
    
    score = (top1_hit * 0.5) + (topic_hit * 0.3) + (skill_hit * 0.2)
    return round(score, 4)


def compute_answer_completeness(answer, ground_truth):
    """How much of the ground truth key info is covered in the answer?"""
    # Extract key terms from ground truth
    import re
    gt_tokens = set(re.findall(r'[a-z0-9]+', ground_truth.lower()))
    ans_tokens = set(re.findall(r'[a-z0-9]+', answer.lower()))
    
    # Remove common stopwords
    stopwords = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
        'and', 'but', 'or', 'not', 'it', 'its', 'this', 'that',
    }
    gt_tokens -= stopwords
    ans_tokens -= stopwords
    
    if not gt_tokens:
        return 1.0
    
    overlap = gt_tokens & ans_tokens
    recall = len(overlap) / len(gt_tokens)
    
    return round(min(1.0, recall * 1.5), 4)  # Scale up slightly since exact match is strict


# ─── RAGAS Evaluation with LLM-based metrics ────────────────────────

def compute_ragas_metrics(eval_results, api_key):
    """
    Compute RAGAS metrics using the ragas library.
    Falls back to custom metrics if ragas fails.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from ragas import EvaluationDataset, SingleTurnSample
        import os
        
        os.environ["OPENAI_API_KEY"] = api_key
        
        # Build RAGAS dataset
        samples = []
        for item in eval_results:
            retrieved_contexts = [doc.get('content', '') for doc in item.get('retrieved_docs_full', [])]
            if not retrieved_contexts:
                retrieved_contexts = ["No context retrieved"]
            
            sample = SingleTurnSample(
                user_input=item['question'],
                response=item['answer'],
                retrieved_contexts=retrieved_contexts,
                reference=item['ground_truth'],
            )
            samples.append(sample)
        
        eval_dataset = EvaluationDataset(samples=samples)
        
        # Run RAGAS evaluation
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        results = evaluate(dataset=eval_dataset, metrics=metrics)
        
        return {
            'faithfulness': round(results['faithfulness'], 4),
            'answer_relevancy': round(results['answer_relevancy'], 4),
            'context_precision': round(results['context_precision'], 4),
            'context_recall': round(results['context_recall'], 4),
            'method': 'ragas_library',
        }
    
    except Exception as e:
        print(f"RAGAS library evaluation failed: {e}")
        print("Falling back to custom metrics...")
        
        # Fallback: compute custom approximations
        faithfulness_scores = []
        relevancy_scores = []
        precision_scores = []
        recall_scores = []
        
        for item in eval_results:
            # Faithfulness approximation: keyword overlap between answer and context
            context_text = ' '.join(doc.get('content', '') for doc in item.get('retrieved_docs_full', []))
            if context_text:
                import re
                ctx_words = set(re.findall(r'[a-z0-9]+', context_text.lower()))
                ans_words = set(re.findall(r'[a-z0-9]+', item['answer'].lower()))
                stopwords = {'the','a','an','is','are','to','of','in','for','on','with','and','but','or','it','this','that'}
                ctx_words -= stopwords
                ans_words -= stopwords
                if ans_words:
                    faith = len(ans_words & ctx_words) / len(ans_words)
                    faithfulness_scores.append(min(1.0, faith * 1.3))
            
            # Answer relevancy: keyword overlap between answer and question
            import re
            q_words = set(re.findall(r'[a-z0-9]+', item['question'].lower()))
            a_words = set(re.findall(r'[a-z0-9]+', item['answer'].lower()))
            q_words -= {'the','a','an','is','are','to','of','in','for','on','with','and','but','or','what','how','why','does','my','do','i'}
            if q_words:
                rel = len(q_words & a_words) / len(q_words)
                relevancy_scores.append(min(1.0, rel * 1.5))
            
            # Context precision
            precision_scores.append(item.get('context_hit_rate', 0.5))
            
            # Context recall
            recall_scores.append(item.get('answer_completeness', 0.5))
        
        return {
            'faithfulness': round(sum(faithfulness_scores) / max(1, len(faithfulness_scores)), 4),
            'answer_relevancy': round(sum(relevancy_scores) / max(1, len(relevancy_scores)), 4),
            'context_precision': round(sum(precision_scores) / max(1, len(precision_scores)), 4),
            'context_recall': round(sum(recall_scores) / max(1, len(recall_scores)), 4),
            'method': 'custom_approximation',
        }


# ─── Main Evaluation Pipeline ──────────────────────────────────────

def run_evaluation(api_key, model="gpt-4o-mini", num_questions=3, verbose=True):
    """Run full evaluation pipeline on the test dataset."""
    
    eval_data = EVAL_DATASET[:num_questions]
    
    if verbose:
        print("=" * 60)
        print("ASK SKILLVERI — RAGAS EVALUATION")
        print("=" * 60)
        print(f"Test questions: {len(eval_data)}")
        print(f"Model: {model}")
        print()
    
    # Initialize components
    kb_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           'knowledge_base', 'domain_knowledge.json')
    documents = load_knowledge_base(kb_path)
    retriever = TFIDFRetriever()
    retriever.build_index(documents)
    
    eval_results = []
    
    for i, test_case in enumerate(eval_data):
        if verbose:
            print(f"\n[{i+1}/{len(eval_data)}] {test_case['question']}")
        
        # Step 1: Retrieve
        retrieved = retriever.retrieve(test_case['question'], top_k=3)
        
        # Step 2: Generate answer
        try:
            answer, tokens = generate_answer(
                test_case['question'], retrieved, api_key, model
            )
            error = None
        except Exception as e:
            answer = f"Error: {str(e)}"
            tokens = 0
            error = str(e)
        
        # Step 3: Compute custom metrics
        context_hit = compute_context_hit_rate(
            retrieved, test_case['expected_skill'], test_case['expected_topic']
        )
        param_mention = compute_parameter_mention_score(answer)
        actionability = compute_actionability_score(answer)
        completeness = compute_answer_completeness(answer, test_case['ground_truth'])
        
        result = {
            'question': test_case['question'],
            'ground_truth': test_case['ground_truth'],
            'answer': answer,
            'expected_skill': test_case['expected_skill'],
            'expected_topic': test_case['expected_topic'],
            'difficulty': test_case['difficulty'],
            'sources_retrieved': [
                {'skill': d['skill'], 'topic': d['topic'], 'relevance': d['relevance_score']}
                for d in retrieved
            ],
            'retrieved_docs_full': retrieved,
            'context_hit_rate': context_hit,
            'parameter_mention': param_mention,
            'actionability': actionability,
            'answer_completeness': completeness,
            'tokens_used': tokens,
            'error': error,
        }
        
        eval_results.append(result)
        
        if verbose:
            print(f"   Context Hit: {context_hit:.2f} | Param Mention: {param_mention:.2f} | "
                  f"Actionability: {actionability:.2f} | Completeness: {completeness:.2f}")
        
        # Small delay to avoid rate limiting
        time.sleep(0.5)
    
    # Step 4: Compute RAGAS aggregate metrics
    if verbose:
        print("\n\nComputing RAGAS aggregate metrics...")
    
    ragas_scores = compute_ragas_metrics(eval_results, api_key)
    
    # Step 5: Compute custom aggregate metrics
    custom_scores = {
        'context_hit_rate': round(
            sum(r['context_hit_rate'] for r in eval_results) / len(eval_results), 4
        ),
        'parameter_mention': round(
            sum(r['parameter_mention'] for r in eval_results) / len(eval_results), 4
        ),
        'actionability': round(
            sum(r['actionability'] for r in eval_results) / len(eval_results), 4
        ),
        'answer_completeness': round(
            sum(r['answer_completeness'] for r in eval_results) / len(eval_results), 4
        ),
    }
    
    # Step 6: Build final report
    report = {
        'timestamp': datetime.now().isoformat(),
        'model': model,
        'num_questions': len(eval_data),
        'ragas_metrics': ragas_scores,
        'custom_metrics': custom_scores,
        'per_question_results': eval_results,
        'total_tokens': sum(r['tokens_used'] for r in eval_results),
    }
    
    # Print summary
    if verbose:
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"\n📊 RAGAS Metrics ({ragas_scores['method']}):")
        print(f"   Faithfulness:      {ragas_scores['faithfulness']:.3f}")
        print(f"   Answer Relevancy:  {ragas_scores['answer_relevancy']:.3f}")
        print(f"   Context Precision: {ragas_scores['context_precision']:.3f}")
        print(f"   Context Recall:    {ragas_scores['context_recall']:.3f}")
        print(f"\n🎯 Custom Skillveri Metrics:")
        print(f"   Context Hit Rate:     {custom_scores['context_hit_rate']:.3f}")
        print(f"   Parameter Mention:    {custom_scores['parameter_mention']:.3f}")
        print(f"   Actionability:        {custom_scores['actionability']:.3f}")
        print(f"   Answer Completeness:  {custom_scores['answer_completeness']:.3f}")
        print(f"\n💰 Total tokens used: {report['total_tokens']:,}")
    
    # Save report
    report_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'evaluation')
    os.makedirs(report_dir, exist_ok=True)
    
    # Save without full doc content to reduce file size
    save_report = report.copy()
    for item in save_report['per_question_results']:
        item.pop('retrieved_docs_full', None)
    
    report_path = os.path.join(report_dir, 'ragas_report.json')
    with open(report_path, 'w') as f:
        json.dump(save_report, f, indent=2, default=str)
    
    if verbose:
        print(f"\n📁 Report saved to {report_path}")
    
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run RAGAS evaluation')
    parser.add_argument('--api-key', type=str, help='OpenAI API key')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='Model to use')
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    
    if not api_key:
        print("ERROR: Please provide an OpenAI API key via --api-key or OPENAI_API_KEY env variable")
        print("Usage: python src/ragas_eval.py --api-key sk-...")
        sys.exit(1)
    
    run_evaluation(api_key, args.model)