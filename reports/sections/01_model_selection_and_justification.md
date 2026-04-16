# Model Selection and Justification

## Design goal

The tutor is deliberately a hybrid NLP system rather than a single end-to-end neural model. The main design trade-off is between:

- explainability for learner feedback and academic analysis
- enough semantic flexibility to handle short paraphrastic answers
- low enough computational cost to run as an interactive tutoring pipeline

This is why classical models are used where transparent lexical evidence is useful, while transformer features are reserved for the parts of the task that genuinely require semantic abstraction.

## Component-level justification

| Component | Chosen approach | Why this was selected | Alternative considered | Why the alternative was not the main choice |
|---|---|---|---|---|
| Language identification | Character TF-IDF + Logistic Regression | Short learner answers expose strong character-level language cues. The model is fast, easy to inspect, and performs strongly on the deployed EN/ES/PL subset. | Fine-tuned BERT classifier | Higher compute cost, less interpretable probabilities, and unnecessary complexity for short-form language ID where char n-grams are already very competitive. |
| Semantic similarity | TF-IDF cosine plus multilingual MiniLM embeddings | TF-IDF gives transparent lexical overlap, while the transformer rescues valid paraphrases and cross-lexical matches. This supports a conservative but explainable tutor policy. | Cross-encoder or fully fine-tuned transformer ranker | Better semantic precision is possible, but inference cost and training-data demands are much higher than the current lesson-bank scale supports. |
| Syntax checking | spaCy parsing with rule-based heuristics | The goal is not full grammar correction. The tutor only needs lightweight, local feedback such as missing verbs, fragments, and overly long sentences. Rules make those decisions easy to audit. | Learned grammatical error detection or full dependency-scoring pipeline | Would require annotated learner-error data and would make error explanations less transparent for a small project. |
| NER and noun phrases | Off-the-shelf spaCy models | Gives contextual signals with minimal engineering and works well as supporting evidence rather than a grading signal. | Custom task-specific NER | The lesson bank is too small to justify training a dedicated entity model. |
| Fluency scoring | n-gram language model with calibrated perplexity bands | Cheap to run, easy to calibrate, and suitable for a "rough naturalness" signal rather than a full proficiency estimate. | Neural LM or LLM-based fluency scorer | More expressive, but much less explainable and harder to control for deterministic feedback. |
| Final tutor decision | Rule-based dialogue policy | Keeps outcomes stable and inspectable. A marker can see exactly why the system returned `GOOD`, `SYNTAX_FIX`, or `MEANING_MISMATCH`. | Learned policy or RL-style reward model | No labelled pedagogical action dataset is available, so a learned policy would add complexity without reliable supervision. |

## Why TF-IDF + Logistic Regression instead of fine-tuned BERT for language ID

- The deployed answers are short, often under a sentence, so character patterns carry most of the useful signal.
- TF-IDF + Logistic Regression is effectively instant at inference time and exposes top probabilities clearly.
- A BERT classifier would increase latency and engineering complexity without solving the main bottleneck in this project.
- The actual difficulty in the tutor is not coarse language ID, but semantic grading and error analysis.

In other words, the project spends model complexity where it matters most: semantic understanding rather than basic language discrimination.

## Why Sentence-BERT style embeddings are used for similarity

- Lexical overlap alone is too brittle for tutoring because learners often use reordering, synonymy, or partial paraphrase.
- Multilingual MiniLM embeddings provide a useful semantic backstop when TF-IDF under-matches valid meaning.
- The transformer is therefore used as a semantic analyser, not as the sole grading signal.

This is a deliberate trade-off: the policy can stay conservative with TF-IDF thresholds while the transformer remains available for analysis, retrieval, and future threshold calibration.

## Why syntax is rule-based instead of full grammar scoring

- The project only needs targeted instructional feedback, not a full grammaticality judgment.
- Rule-based syntax flags map directly to learner-facing advice such as "add a main verb" or "this is a fragment".
- A richer grammar model would be harder to explain, harder to validate on a small dataset, and harder to debug when it disagrees with human judgment.

## Critical limitations

- The lesson bank is small and handcrafted, so coverage is narrow and generalisation is limited.
- TF-IDF is conservative: it protects against false positives, but it also rejects some valid paraphrases.
- Transformer similarity is more flexible, but it can over-score partial answers, so it cannot be trusted alone for grading.
- Syntax decisions depend on spaCy parsing quality, which makes them sensitive to accent removal, fragments, and unusual word order.
- Fluency scoring is only a heuristic. n-gram perplexity is useful for relative ranking, but not for measuring proficiency in any rigorous sense.
- Threshold-based policies are easy to explain, but they are also sensitive to calibration choices. A threshold that works for exact lesson targets may be too strict for real learner variation.

## Overall justification

The system is best understood as an explainable tutoring pipeline rather than a leaderboard model. Classical models are used where interpretability and stability are most important. Transformer features are introduced only where lexical models are insufficient. That hybrid design is the main architectural justification for the project.
