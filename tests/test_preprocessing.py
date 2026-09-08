from biogenesis.preprocessing.chunker import chunk_paper, split_sentences
from biogenesis.preprocessing.cleaner import clean_text


def test_clean_text_strips_whitespace():
    assert clean_text("  hello   world  ") == "hello world"


def test_clean_text_strips_copyright():
    text = "This is the real finding. Copyright 2020 Elsevier. All rights reserved."
    cleaned = clean_text(text)
    assert "Copyright" not in cleaned
    assert "real finding" in cleaned


def test_split_sentences_basic():
    text = "First sentence. Second sentence! Third one?"
    sentences = split_sentences(text)
    assert len(sentences) == 3


def test_chunk_paper_includes_title_and_abstract_chunks():
    chunks = chunk_paper(
        pmid="123",
        title="A study on X",
        abstract="Sentence one. Sentence two. Sentence three. Sentence four.",
        chunk_sentences=2,
        overlap=1,
    )
    assert any(c.source_field == "title" for c in chunks)
    assert any(c.source_field == "abstract" for c in chunks)
    assert all(c.pmid == "123" for c in chunks)
