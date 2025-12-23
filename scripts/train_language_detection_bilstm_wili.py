from nlp_tutor.datasets.wili import load_wili_2018
from nlp_tutor.classification.lang_detect_bilstm_chars import train_eval_save, predict_language, strip_diacritics, add_typo_noise

if __name__ == "__main__":
    ds = load_wili_2018("train")

    r = train_eval_save(
        ds=ds,
        target_langs={"English", "Spanish", "Polish"},
        max_chars=120,      # give the LSTM enough sequence
        epochs=6,
        batch_size=64,
        lr=2e-3,
    )

    print("\nFinal metrics:")
    print(f"Accuracy:  {r.accuracy:.4f}")
    print(f"F1-macro:  {r.f1_macro:.4f}")
    print("\nReport:\n")
    print(r.report)

    print("\nShort-text checks (BiLSTM):")
    tests = ["yes", "no", "ok", "thanks", "sí", "gracias", "hola", "tak", "nie", "dzień dobry", "przepraszam", "co tam"]
    for t in tests:
        print(f"{t!r:18} -> {predict_language(t)}")

    print("\nNo-diacritics checks (realistic typing):")
    for t in ["sí", "gracias", "dzień dobry", "przepraszam"]:
        t2 = strip_diacritics(t)
        print(f"{t2!r:18} -> {predict_language(t2)}")

    print("\nLight typo-noise checks:")
    for t in ["gracias", "dzień dobry", "przepraszam", "thanks"]:
        t2 = add_typo_noise(t, p=0.10)
        print(f"{t2!r:18} -> {predict_language(t2)}")
