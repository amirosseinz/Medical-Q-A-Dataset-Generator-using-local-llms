import random

import pytest

from app.services.export_service import QAPairForExport, split_dataset


def row(name, *sources, legacy=False):
    passages = [{"source_id": source} for source in sources]
    metadata = {"citations": passages} if legacy else {"retrieved_context": {"passages": passages}}
    return QAPairForExport(name, "Answer", metadata=metadata)


def test_sources_and_transitive_links_are_indivisible():
    pairs = [row("a", "A"), row("bc", "B", "C"), row("ab", "A", "B"),
             row("c", "C", legacy=True), row("d", "D"), row("e", "E"), row("f", "F")]
    result = split_dataset(pairs, 0.5, 0.25, 0.25, split_strategy="source")
    locations = {pair.question: name for name, rows in result.items() for pair in rows}
    assert len(locations) == len(pairs)
    assert len({locations[name] for name in ("a", "bc", "ab", "c")}) == 1
    assert sorted(id(pair) for rows in result.values() for pair in rows) == sorted(map(id, pairs))
    assert result == split_dataset(pairs, 0.5, 0.25, 0.25, split_strategy="source")


def test_zero_fraction_and_single_large_component():
    pairs = [row(str(i), "A") for i in range(20)]
    result = split_dataset(pairs, 0.8, 0, 0.2, split_strategy="source")
    assert result == {"train": pairs, "val": [], "test": []}
    assert split_dataset([], split_strategy="source") == {"train": [], "val": [], "test": []}


@pytest.mark.parametrize("metadata", [None, {}, {"citations": []}, {"citations": [{}]},
    {"citations": [{"source_id": " "}]}, {"citations": [{"source_id": "a"}, {}]},
    {"retrieved_context": {"passages": []}, "citations": [{"source_id": "a"}]}])
def test_incomplete_provenance_is_rejected(metadata):
    with pytest.raises(ValueError, match="source ID"):
        split_dataset([QAPairForExport("Q", "A", metadata=metadata)], split_strategy="source")


@pytest.mark.parametrize("fractions", [(0.8, 0.2, 0.2), (-0.1, 0.1, 1), (float("nan"), 0, 1)])
def test_invalid_fractions_are_rejected(fractions):
    with pytest.raises(ValueError, match="fractions"):
        split_dataset([row("a", "A")], *fractions, split_strategy="source")


def test_row_split_keeps_legacy_assignment_and_local_random_state():
    pairs = [row(str(i), str(i)) for i in range(20)]
    expected = list(pairs)
    random.Random(42).shuffle(expected)
    before = random.getstate()
    result = split_dataset(pairs)
    assert result == {"train": expected[:16], "val": expected[16:18], "test": expected[18:]}
    assert random.getstate() == before
