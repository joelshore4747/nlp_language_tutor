# CMU657 Natural Language Processing — Project  
**LinguaTrail: An Adaptive NLP Tutor Pipeline (Language ID, Semantics, Syntax, NER, Fluency)**  

**Author:** Joel Shore  
**Module:** CMU657 Natural Language Processing  
**Submission Date:** 16/12/2025  

---

## Abstract  

The project is a compact NLP tutoring system designed to give language learners fast, structured feedback on short translation or comprehension answers. My main goal wasn’t to build another “right or wrong” quiz, but something that felt closer to a real tutor, a system that could tell what the learner meant, not just whether they used the exact same words at least this was the idea.  

To do this, I built a pipeline combining five key NLP components: **language identification**, **semantic similarity**, **syntax heuristics**, **named entity recognition (NER)**, and **fluency scoring**. Each one provides a specific signal about the learner’s response. Together, they form an explainable tutor system that can return clear outcomes like **GOOD**, **RETRY**, or **LANGUAGE MISMATCH**.  

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

---

## 5. Evaluation  

### Language identification  

Language identification was evaluated using 1,500 WiLI samples (500 per language).  

- TF IDF model: **0.9927 accuracy**  
- BiLSTM model: **0.9440 accuracy**  

This supports an important takeaway: for short text language ID, strong classical baselines can be extremely competitive.  

### Semantic similarity  

Semantic similarity was tested using lesson bank sentences against random negatives. Both TF IDF and transformer embeddings separated positives and negatives clearly, showing the pipeline logic works, although real paraphrase coverage is still limited by lesson bank scope.  

A supporting Jupyter notebook demonstrates predictions, nearest neighbour retrieval, syntax and NER examples, and policy execution in a reproducible way.  

---

## 6. Discussion and Improvements  

The main strength of LinguaTrail is that it feels like a real tutor system rather than a single model demo. It integrates multiple NLP skills, separates backend logic from UI, and gives feedback that can be explained.  

The key limitation is scope: the lesson bank is small, semantic evaluation is most reliable for near exact matches, and fluency signals can degrade on very short or mixed language text.  

If I continued the project, the biggest improvements would be:  

- Expanding the lesson bank with paraphrases and learner error patterns  
- Calibrating thresholds using real learner data  
- Training stronger per language fluency models  
- Exploring controlled fine tuning of transformer embeddings  
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

---

## Appendix B — Notebook Summary (NLP Models and Functions Analysis)  

The notebook includes lightweight reproducible checks for the tutor pipeline and runs entirely without raw datasets. It validates that the packaged models and lesson resources operate correctly, and that the tutor pipeline produces stable outputs even when optional dependencies are missing.  

---

## References (Newman–Harvard Style)  

Thoma, M. (2018) WiLI 2018: Wikipedia Language Identification dataset. Available at: https://zenodo.org/records/840106 (Accessed: 16 December 2025).  

Maas, A.L., Daly, R.E., Pham, P.T., Huang, D., Ng, A.Y. and Potts, C. (2011) Learning word vectors for sentiment analysis. Available at: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews (Accessed: 16 December 2025).  

Reimers, N. and Gurevych, I. (2019) ‘Sentence BERT: Sentence embeddings using Siamese BERT networks’, arXiv. Available at: https://arxiv.org/abs/1908.10084 (Accessed: 16 December 2025).  

Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C. and others (2020) ‘Transformers: State of the art natural language processing’, Proceedings of EMNLP: System Demonstrations. Available at: https://huggingface.co/transformers/ (Accessed: 16 December 2025).  

spaCy (2024) Industrial strength Natural Language Processing in Python. Available at: https://spacy.io/ (Accessed: 16 December 2025).  

NLTK Project (2024) Natural Language Toolkit. Available at: https://www.nltk.org/ (Accessed: 16 December 2025).  

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V. and others (2011) ‘Scikit learn: machine learning in Python’, Journal of Machine Learning Research, 12, pp. 2825–2830. Available at: https://scikit-learn.org/ (Accessed: 16 December 2025).  

Paszke, A., Gross, S., Massa, F., Lerer, A. and others (2019) ‘PyTorch: An imperative style, high performance deep learning library’, Advances in Neural Information Processing Systems (NeurIPS). Available at: https://pytorch.org/ (Accessed: 16 December 2025).  

FastAPI (2025) FastAPI Documentation. Available at: https://fastapi.tiangolo.com/ (Accessed: 16 December 2025).  

SolidJS (2025) SolidJS Documentation. Available at: https://www.solidjs.com/ (Accessed: 16 December 2025).  

Module lecture materials (2025) Natural Language Processing (CMU657) slides and lab notes via Moodle. Newman University. (Accessed: 16 December 2025).  
