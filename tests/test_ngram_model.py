from nlp_tutor.ngram_model import NgramLanguageModel, NgramLMConfig


def test_perplexity_direction_better_order_lower():
    lm = NgramLanguageModel(NgramLMConfig(order=2, add_k=1.0))

    corpus = []
    for _ in range(80):
        corpus.append(["i", "like", "apples"])
        corpus.append(["you", "like", "apples"])
        corpus.append(["i", "really", "like", "apples"])
    lm.fit(corpus)

    good = ["i", "like", "apples"]
    bad = ["apples", "like", "i"]

    ppl_good = lm.perplexity(good)
    ppl_bad = lm.perplexity(bad)

    assert ppl_good < ppl_bad, f"Expected good order to be more fluent. good={ppl_good}, bad={ppl_bad}"
