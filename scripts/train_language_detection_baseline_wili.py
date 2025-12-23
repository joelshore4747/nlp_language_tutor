from nlp_tutor.datasets.wili import load_wili_2018
from nlp_tutor.classification.lang_detect_baseline import train_eval_save, load_model, short_text_sanity_check, \
    predict_language_with_confidence

if __name__ == "__main__":
    ds = load_wili_2018("train")

    r = train_eval_save(
        ds=ds,
        target_langs={"English", "Spanish", "Polish"},
        max_chars=50,  # realistic “short message” setting
    )

    print(f"\nAccuracy: {r.accuracy:.4f}")
    print(f"F1-macro: {r.f1_macro:.4f}")
    print("\nReport:\n")
    print(r.report)

    model = load_model()
    print("\nShort-text sanity checks:")
    for text, pred in short_text_sanity_check(model):
        print(f"{text!r:18} -> {pred}")

    for t in ["yes", "no", "ok", "sí", "hola", "tak", "dzień dobry", "co tam"]:
        print(t, predict_language_with_confidence(t, model=model, threshold=0.70))
