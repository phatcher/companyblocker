from company_vectorize import HashVectorizer, l2_normalize


def test_hash_vectorizer_is_deterministic_and_dimensioned():
    vectorizer = HashVectorizer(dimension=16)

    a = vectorizer.embed_texts(["Acme Ltd"])[0]
    b = vectorizer.embed_texts(["Acme Ltd"])[0]

    assert len(a) == 16
    assert a == b


def test_l2_normalize_zero_vector_returns_copy():
    source = [0.0, 0.0, 0.0]
    actual = l2_normalize(source)

    assert actual == source
    assert actual is not source
