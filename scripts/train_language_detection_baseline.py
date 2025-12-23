from __future__ import annotations

from nlp_tutor.classification.lang_detect_baseline import train_evaluate_save

if __name__ == "__main__":
    r = train_evaluate_save()
    print("\nSaved model to data/models/")
    print(f"Accuracy: {r.accuracy:.4f}")
    print(f"F1-macro: {r.f1_macro:.4f}")
    print("\nClassification report:\n")
    print(r.report)
