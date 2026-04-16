# Evaluation, Ablation, and Failure Analysis

## Evaluation design

The added evaluation work is split into three parts:

1. a quantitative language-ID evaluation on the deployed EN/ES/PL subset of WiLI 2018
2. a small semantic-threshold benchmark over exact matches, paraphrases, partial answers, and mismatches
3. a functional tutor benchmark with ablations for syntax and fluency

All supporting CSV and JSON outputs are included in `reports/analysis/`.

## 1. Language identification

The deployed baseline model was evaluated on 1,500 held-out WiLI samples: 500 English, 500 Spanish, and 500 Polish.

- Accuracy: `0.9860`
- Macro F1: `0.9860`

### Confusion matrix

| True \ Pred | English | Polish | Spanish |
|---|---:|---:|---:|
| English | 497 | 2 | 1 |
| Polish | 5 | 493 | 2 |
| Spanish | 6 | 5 | 489 |

### Per-language precision and recall

| Language | Precision | Recall | F1 |
|---|---:|---:|---:|
| English | 0.9783 | 0.9940 | 0.9861 |
| Polish | 0.9860 | 0.9860 | 0.9860 |
| Spanish | 0.9939 | 0.9780 | 0.9859 |

### Interpretation

- The baseline is strong enough that language ID is not the project bottleneck.
- Errors are small and distributed rather than dominated by one class collapsing into another.
- This supports the decision to deploy the classical classifier instead of replacing it with a much heavier transformer classifier.

Related screenshots for this subsection:

- `reports/images/confusion_matrix.png`
- `reports/images/precision_recall_f1.png`

## 2. Semantic threshold study

I created a small semantic benchmark with 12 cases covering:

- exact answers
- word-order variants
- lexical paraphrases
- partial answers
- clearly wrong answers

### Mean similarity by case type

| Label | Mean TF-IDF | Mean transformer |
|---|---:|---:|
| match | 0.5579 | 0.8897 |
| partial | 0.4421 | 0.8275 |
| mismatch | 0.0455 | 0.2314 |

### What this shows

- TF-IDF separates clear mismatches well, but it is conservative even on valid paraphrases.
- The transformer is much better at recognising semantic closeness across rewording and accent loss.
- The transformer also scores some partial answers quite highly, which means it is not safe to use alone as the grading rule.

This is a useful architectural result: the project is justified as a hybrid system because each similarity signal fails differently.

### Threshold examples

- `Donde esta la estacion de tren?` vs target question:
  TF-IDF `0.2244`, transformer `0.9793`
  This is a near-exact meaning match that lexical overlap underserves.
- `Quiero un vaso con agua por favor.` vs `Quiero un vaso de agua, por favor.`:
  TF-IDF `0.6279`, transformer `0.9766`
  A small paraphrase falls below a strict TF-IDF cutoff.
- `A las siete cada dia.` vs `Me levanto a las siete cada día.`:
  TF-IDF `0.4071`, transformer `0.8590`
  The transformer captures topical relatedness, but the answer is still incomplete.

The main implication is that threshold tuning has to be pedagogical, not only statistical. A high transformer score does not always mean the learner has produced a complete acceptable answer.

Related screenshot for this subsection:

- `reports/images/semantic_threshold.png`

## 3. Functional ablation study

I built a small 8-case tutor benchmark with expected actions such as `GOOD`, `SYNTAX_FIX`, `MEANING_MISMATCH`, and `FLUENCY_NUDGE`.

| Variant | Correct cases | Total | Action match rate |
|---|---:|---:|---:|
| Full system | 8 | 8 | 1.000 |
| No syntax | 6 | 8 | 0.750 |
| No fluency | 7 | 8 | 0.875 |

### Interpretation

- Removing syntax reduces action accuracy most sharply because fragment detection is a core tutor function.
- Removing fluency causes a smaller but still meaningful drop: one benchmark case that should be a style nudge becomes `GOOD`.
- The benchmark is small and manual, so this is not a claim of full generalisation. It is a functional ablation showing that the extra components are doing real work inside the policy.

Related screenshot for this subsection:

- `reports/images/Tutor_ablation_results.png`

## 4. Failure cases

The strongest improvement opportunities come from the failure analysis rather than the benchmark wins.

| Case | Expected | Predicted | Why it failed |
|---|---|---|---|
| `Donde esta la estacion de tren?` | `GOOD` | `SYNTAX_FIX` | Accent-free input hurt the parser, which then hallucinated a missing verb. |
| `Me levanto a las siete cada dia.` | `GOOD` | `MEANING_MISMATCH` | TF-IDF `0.6949` fell below the current `0.80` policy threshold even though transformer similarity was `0.9980`. |
| `A las siete.` | `GOOD` | `INPUT_TOO_SHORT` | The minimum-length gate blocks short but plausible learner answers before semantic scoring runs. |
| `Trabajo en mi tesis este año.` | `GOOD` | `SYNTAX_FIX` | The exact target is sometimes mis-flagged by the syntax heuristic. |
| `Estudio procesamiento del lenguaje natural en la universidad hoy.` | `GOOD` | `SYNTAX_FIX` | Acronym expansion plus parser behaviour makes a valid paraphrase look like a fragment. |

Related screenshot for this subsection:

- `reports/images/tutor_failure_case.png`

## 5. Critical reflection

This extra evaluation strengthens the project in two ways.

First, it demonstrates that the system is not just "working" in a demo sense: the deployed language gate is quantitatively strong, and the extra tutor components measurably change policy outcomes.

Second, it exposes the real weaknesses clearly:

- exact-threshold TF-IDF grading is too strict for some valid learner variation
- parser-based syntax rules are brittle when accents or lexical form change
- the fluency model is useful for nudging style, but it is not a dependable quality score by itself
- very short but acceptable learner answers are currently under-served

Those limitations are precisely the areas that would need improvement to move from a strong project prototype to a more robust educational system.
