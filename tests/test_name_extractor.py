import sys
import types

from ttsforge import name_extractor


def test_spacy_model_cached(monkeypatch) -> None:
    calls = {"count": 0}

    class FakeEnt:
        def __init__(self, text: str, label: str) -> None:
            self.text = text
            self.label_ = label

    class FakeDoc:
        def __init__(self, text: str) -> None:
            self.ents = [FakeEnt("Alice", "PERSON")]

    class FakeNLP:
        def pipe(self, chunks, batch_size=4):
            for chunk in chunks:
                yield FakeDoc(chunk)

    def fake_load(model_name: str):
        calls["count"] += 1
        return FakeNLP()

    fake_spacy = types.SimpleNamespace(load=fake_load)
    monkeypatch.setitem(sys.modules, "spacy", fake_spacy)

    name_extractor._get_nlp.cache_clear()

    text = "Alice went to Wonderland."
    name_extractor.extract_names_from_text(text)
    name_extractor.extract_names_from_text(text)

    assert calls["count"] == 1
