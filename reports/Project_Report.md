# CMU657 Natural Language Processing — Project  
**Language Application: An Adaptive NLP Tutor Pipeline (Language ID, Semantics, Syntax, NER, Fluency)**  

**Author:** Joel Shore  
**Module:** CMU657 Natural Language Processing  
**Submission Date:** 16/12/2025  

---

## Abstract  

The project is a compact NLP tutoring system designed to give language learners fast, structured feedback on short translation or comprehension answers. My main goal wasn’t to build another “right or wrong” quiz, but something that felt closer to a real tutor, a system that could tell what the learner meant, not just whether they used the exact same words at least this was the idea.  

To do this, I built a pipeline combining five key NLP components: **language identification**, **semantic similarity**, **syntax heuristics**, **named entity recognition (NER)**, and **fluency scoring**. Each component contributes a distinct linguistic signal, enabling a structured and interpretable decision process. about the learner’s response. Together, they form an explainable tutor system that can return clear outcomes like **GOOD**, **RETRY**, or **LANGUAGE MISMATCH**.  

The project uses a **FastAPI backend** to run the tutor logic and a **SolidJS frontend** inspired by Duolingo to make it more interactive. While it’s not a full conversational LLM tutor, it demonstrates how classical and neural NLP methods can work together to create a meaningful, interpretable learning tool.  

---

## 1. Introduction and Aim  

When I started this project, I wanted to solve a common frustration in language apps: shallow or inconsistent feedback lacking ability to teach concisely. Most beginner apps simply mark answers right or wrong based on matching words. This often punishes learners who give valid but slightly different answers.  

On the other hand, LLM based tutors can be great at understanding meaning, but they can also be unpredictable — sometimes giving vague explanations or drifting away from lesson goals. For an academic NLP project, I needed something more controlled and measurable.  

So my goal became:  

> Build a structured tutor that evaluates learner responses using multiple NLP signals and provides clear, repeatable feedback that makes sense to the learner.  

The main objectives were:  

- Detect whether the learner responded in the correct language (English, Spanish, or Polish).  
- Evaluate whether the response captures the intended meaning, not just the same words.  
- Provide simple syntax and fluency feedback in a way learners can understand.  
- Extract named entities and key phrases to support richer future lessons.  
- Return an overall tutor outcome such as **GOOD**, **LANGUAGE MISMATCH**, or **RETRY**.  

---

## 2. System Overview  

### Backend — FastAPI  

The backend is built with **FastAPI**, chosen for its speed and easy API structure. Each endpoint focuses on one small, explainable part of the tutor, for example:  

- `/tutor/evaluate` — combines all components and returns the final tutor result.  
- `/semantic/score` — tests semantic similarity between pairs of sentences.  

This modular design means each part of the system can be understood and improved independently. The backend returns a single JSON response containing all signals — language ID, semantic score, syntax flags, fluency level, and extracted entities.  

### Frontend — SolidJS  

The **SolidJS** interface was designed to feel familiar, similar to Duolingo lessons. It presents a lesson prompt and lets the user check their response. What makes it different is the analysis panel — it shows what the tutor actually detected:  

- predicted language  
- semantic similarity score  
- syntax structure comments  
- fluency band (low, medium, high)  
- named entities  

This transparency helps the learner understand why a response was judged a certain way.  

The frontend could be expanded to include voice input later, using speech to text so learners can practise spoken responses with the same analysis pipeline.  

---

## 3. Data and Resources  

The lessons come from a small handcrafted lesson bank I created myself. Each prompt is short (around 6–7 words) and focuses on simple topics like daily routine or travel.  

For the NLP models, I used:  

- **WiLI 2018** — for language identification (English, Spanish, Polish).  
- **IMDB Reviews** — for optional sentiment analysis experiments.  

This small, well defined dataset setup made the project easier to test and reproduce without needing massive raw data files.  

---

## 4. Methods and Components  

### 4.1 Preprocessing  

Responses are normalised — lowercased, cleaned, and stripped of extra whitespace — to ensure consistent results even if the learner types casually.  

### 4.2 Language Identification  

This step checks if the learner answered in the expected language. It uses two models:  

- **TF IDF + Logistic Regression** (classical baseline, highly accurate)  
- **Character level BiLSTM** (neural approach, more flexible)  

Character features work well because short texts show clear language patterns (for example, Spanish accents or Polish characters).  

### 4.3 Semantic Similarity  

This is the “meaning” part of the tutor. It measures how close the learner’s response is to the target sentence:  

- **TF IDF cosine similarity** — simple and interpretable  
- **Sentence BERT embeddings** — better for paraphrases  

Together, they balance speed and realism, showing how well a learner’s meaning aligns with the expected answer.  

### 4.4 Syntax Analysis  

Using **spaCy**, the tutor detects common issues such as missing verbs, fragments, or sentences that are too long. It doesn’t try to score full grammar — just highlight patterns that affect clarity.  

### 4.5 Named Entity Recognition (NER)  

Entities and noun phrases are extracted to provide context. For example, if a learner mentions London or Monday, the tutor can identify that information and, in the future, use it for more detailed feedback.  

### 4.6 Fluency Scoring  

Fluency is estimated using an N gram language model. It measures how natural the sentence looks, using perplexity bands (low, medium, high). This isn’t a perfect fluency test, but it helps show whether a sentence is coherent and readable.  

### 4.7 Dialogue Policy  

All of these components feed into a final decision function called **choose action**. It applies simple but clear rules:  

| Condition | Outcome |
|----------|---------|
| Wrong language | LANGUAGE MISMATCH |
| Low semantic similarity | RETRY |
| Good meaning + acceptable structure | GOOD |

This policy ensures predictable, consistent behaviour — something most AI tutors lack.  

### 4.8 Model Selection and Justification  

The most important design choice in the project is that LinguaTrail is a **hybrid system** rather than a single end to end neural model. That decision was deliberate. I used **classical models** where interpretability and low latency were most important, and **transformer based semantics** only where lexical matching was too brittle.  

- **TF IDF + Logistic Regression** was chosen for language ID because short learner responses carry strong character level language cues, and the model is both fast and easy to inspect. A fine tuned BERT classifier was considered, but it would have increased compute cost and reduced explainability without solving the main challenge in the tutor.  
- **TF IDF + multilingual transformer similarity** was used for semantics because the tutor needs both conservative grading and some ability to recognise paraphrase. TF IDF gives transparent lexical evidence; transformer embeddings provide semantic flexibility when wording changes.  
- **Rule based syntax checks on top of spaCy parsing** were used instead of full grammar scoring because the project only needs targeted learner feedback such as “missing verb” or “fragment”, not a full grammaticality estimate.  
- **N gram fluency scoring** was selected because it is cheap, deterministic, and easy to calibrate into broad bands. A neural language model would be more expressive, but much less explainable.  

This justification matters because it explains the system as a set of deliberate engineering trade offs: **explainability vs flexibility**, **speed vs model complexity**, and **repeatability vs open ended generation**. A fuller critique is provided in `reports/sections/01_model_selection_and_justification.md`.  

---

## 5. Evaluation  

### Language identification  

Language identification was evaluated on 1,500 held out WiLI samples (500 English, 500 Spanish, 500 Polish) using the packaged baseline model actually deployed in the tutor.  

- Baseline char TF IDF + Logistic Regression: **0.9860 accuracy**, **0.9860 macro F1**  

The confusion matrix was:

| True \ Pred | English | Polish | Spanish |
|----------|---------:|--------:|---------:|
| English | 497 | 2 | 1 |
| Polish | 5 | 493 | 2 |
| Spanish | 6 | 5 | 489 |

This supports an important takeaway: for short text language ID, strong classical baselines can be extremely competitive. The screenshot figures from the evaluation notebook are referenced as **Figure A4** (`confusion_matrix.png`) and **Figure A5** (`precision_recall_f1.png`).  

### Semantic similarity  

Semantic similarity was expanded into a small threshold study covering exact matches, paraphrases, partial answers, and wrong answers. The strongest pattern was that **TF IDF is conservative**, while the transformer is **more paraphrase aware but more willing to over score partial answers**. On the semantic benchmark:

| Label | Mean TF IDF | Mean transformer |
|----------|------------:|-----------------:|
| match | 0.5579 | 0.8897 |
| partial | 0.4421 | 0.8275 |
| mismatch | 0.0455 | 0.2314 |

This result justifies the hybrid design. TF IDF protects against false positives, but it misses some valid rewordings. Transformer similarity recovers those cases, but it can also make incomplete answers look too acceptable if used on its own. The notebook screenshot for this comparison is referenced as **Figure A6** (`semantic_threshold.png`).  

### Ablation and failure analysis  

I also added a small functional benchmark for the full tutor policy. On an 8 case action benchmark, the **full system** matched the intended tutor action on **8/8 cases**, while **removing syntax** dropped to **6/8** and **removing fluency** dropped to **7/8**. This shows that the extra components are not decorative; they change decisions in meaningful ways. The ablation screenshot is referenced as **Figure A7** (`Tutor_ablation_results.png`).  

The failure analysis was equally important. Five examples showed the current limits clearly:

- accent free Spanish questions can trigger false syntax errors  
- near exact answers can fall below the current TF IDF policy threshold  
- short but plausible answers can be blocked by the minimum length gate  
- some exact lesson targets are still parser sensitive  
- lexical paraphrases such as acronym expansion remain under served  

These failure patterns are illustrated by **Figure A8** (`tutor_failure_case.png`).  

Supporting material is now split across:

- `reports/sections/02_evaluation_and_failure_analysis.md`
- `reports/analysis/`
- `reports/notebooks/01_nlp_models_analysis.ipynb`
- `reports/notebooks/02_language_id_evaluation_report.ipynb`
- `reports/notebooks/03_tutor_pipeline_evaluation_report.ipynb`

---

## 6. Discussion and Improvements  

The main strength of LinguaTrail is that it feels like a real tutor system rather than a single model demo. It integrates multiple NLP skills, separates backend logic from UI, and gives feedback that can be explained.  

The expanded evaluation also makes the limitations clearer. The lesson bank is still small and handcrafted, so coverage is narrow. TF IDF based grading is robust against obvious mismatches, but can be too strict for valid paraphrases. Transformer similarity is more flexible, but can over accept partial answers. Syntax rules are useful for fragments, yet brittle when accents are removed or parsing becomes unstable. Fluency scoring helps produce a more tutor like response, but it should be treated as a heuristic rather than a true proficiency score.  

If I continued the project, the biggest improvements would be:  

- Expanding the lesson bank with paraphrases and learner error patterns  
- Calibrating thresholds using real learner data  
- Training stronger per language fluency models  
- Exploring controlled fine tuning or calibration of transformer similarity  
- Extending the interface to voice input so learners can speak answers naturally  

---

## 7. Conclusion  

LinguaTrail delivers a working, explainable NLP tutoring pipeline that evaluates learner responses beyond simple word matching. By combining language identification, semantic similarity, syntax heuristics, NER, and fluency scoring into one policy driven decision, it demonstrates how applied NLP can be turned into a realistic learning tool.  

While not a full conversational LLM tutor, it achieves the project’s original goal: a controlled tutor style app that can give meaningful feedback based on multiple linguistic signals, and a strong foundation for future expansion into speech based interaction and richer learner modelling.  

---

## Appendix A — Figures and Screenshots  

**Figure A1 — ui landing page.png**  
LinguaTrail landing view showing lesson path and analysis panels before evaluation.  

**Figure A2 — ui language mismatch.png**  
Example evaluation showing LANGUAGE MISMATCH when the detected language differs from the expected one.  

**Figure A3 — ui success good.png**  
Successful evaluation returning GOOD with semantic scores, syntax status, fluency output and entity extraction.  

**Figure A4 — confusion_matrix.png**  
Language identification confusion matrix from the evaluation notebook, showing a strong diagonal and limited cross language confusion on the EN/ES/PL WiLI subset.  

**Figure A5 — precision_recall_f1.png**  
Per language precision, recall, and F1 summary for the deployed baseline language identifier.  

**Figure A6 — semantic_threshold.png**  
Semantic threshold comparison showing how TF IDF and transformer similarity behave across matches, partial answers, and mismatches.  

**Figure A7 — Tutor_ablation_results.png**  
Ablation summary comparing the full tutor policy against variants without syntax or fluency signals.  

**Figure A8 — tutor_failure_case.png**  
Failure case screenshot highlighting where the current tutor still misclassifies or over penalises valid learner input.  

---

## Appendix B — Notebook Summary (NLP Models and Functions Analysis)  

Three notebooks now support the report:

- `01_nlp_models_analysis.ipynb` — lightweight checks of the packaged tutor components  
- `02_language_id_evaluation_report.ipynb` — confusion matrix and per language precision/recall views  
- `03_tutor_pipeline_evaluation_report.ipynb` — semantic threshold tables, ablation results, and failure cases  

Together they provide screenshot ready views of the data behind the written report sections.  

---

## References (Newman–Harvard Style)  

Thoma, M. (2018) WiLI 2018: Wikipedia Language Identification dataset. Available at: https://zenodo.org/records/840106 (Accessed: 16 December 2025).  

Mexwell (n.d.) WiLI-2018 Dataset. Kaggle. Available at: https://www.kaggle.com/datasets/mexwell/wili-2018 (Accessed: 2 February 2026).  

Lakshmipathi, N. (2019) IMDB Dataset of 50K Movie Reviews. Kaggle. Available at: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews (Accessed: 2 February 2026).  

Reimers, N. and Gurevych, I. (2019) ‘Sentence BERT: Sentence embeddings using Siamese BERT networks’, arXiv. Available at: https://arxiv.org/abs/1908.10084 (Accessed: 16 December 2025).  

Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C. and others (2020) ‘Transformers: State of the art natural language processing’, Proceedings of EMNLP: System Demonstrations. Available at: https://huggingface.co/transformers/ (Accessed: 16 December 2025).  

spaCy (2024) Industrial strength Natural Language Processing in Python. Available at: https://spacy.io/ (Accessed: 16 December 2025).  

NLTK Project (2024) Natural Language Toolkit. Available at: https://www.nltk.org/ (Accessed: 16 December 2025).  

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V. and others (2011) ‘Scikit learn: machine learning in Python’, Journal of Machine Learning Research, 12, pp. 2825–2830. Available at: https://scikit-learn.org/ (Accessed: 16 December 2025).  

Paszke, A., Gross, S., Massa, F., Lerer, A. and others (2019) ‘PyTorch: An imperative style, high performance deep learning library’, Advances in Neural Information Processing Systems (NeurIPS). Available at: https://pytorch.org/ (Accessed: 16 December 2025).  

FastAPI (2025) FastAPI Documentation. Available at: https://fastapi.tiangolo.com/ (Accessed: 16 December 2025).  

SolidJS (2025) SolidJS Documentation. Available at: https://www.solidjs.com/ (Accessed: 16 December 2025).  

Module lecture materials (2025) Natural Language Processing (CMU657) slides and lab notes via Moodle. Newman University. (Accessed: 16 December 2025).  
