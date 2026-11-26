import numpy as np
import pytest

from embedstamp import (
    EmbedstampError,
    FakeEmbedder,
    LocalIndex,
    SwitchRefusedError,
    migrate,
    stamp,
    verify,
)
from embedstamp.demo import load_lines
from embedstamp.stores.chroma import ChromaStore

DOCS = load_lines("demo_corpus.txt")[:60]
IDS = [f"doc-{i:03d}" for i in range(len(DOCS))]


def make_local(tmp_path):
    idx = LocalIndex(tmp_path / "docs")
    embed = FakeEmbedder("fake:a")
    idx.add(IDS, DOCS, embed(DOCS))
    idx.save(stamp(embed, model="fake:a"))
    return idx


def make_chroma(tmp_path, chromadb, settings):
    client = chromadb.PersistentClient(path=str(tmp_path / "db"), settings=settings)
    col = client.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"}, embedding_function=None)
    embed = FakeEmbedder("fake:a")
    col.add(ids=IDS, documents=DOCS, embeddings=embed(DOCS).tolist())
    store = ChromaStore(col, client)
    store.save(stamp(embed, model="fake:a"))
    return store


# local 은 항상 돌고, chroma 는 chroma 마커가 붙어 chromadb 가 없으면 건너뛴다(CI 필수 잡에서도 빠진다)
@pytest.fixture(params=["local", pytest.param("chroma", marks=pytest.mark.chroma)])
def store(request, tmp_path):
    if request.param == "local":
        return make_local(tmp_path)
    # chroma 일 때만 chromadb 픽스처를 부른다 — local 매개변수는 chromadb 없이도 돈다
    chromadb = request.getfixturevalue("chromadb")
    return make_chroma(tmp_path, chromadb, request.getfixturevalue("chroma_settings"))


def names(store):
    if isinstance(store, LocalIndex):
        return {p.name for p in store.path.parent.iterdir()}
    return {c.name for c in store._client.list_collections()}


def test_report_without_switch_keeps_blue_live(store):
    new, old = FakeEmbedder("fake:b"), FakeEmbedder("fake:a")
    report = migrate(store, new, new_model="fake:b", old_embed=old, k=10)
    assert report.n_docs == len(DOCS) and report.n_embedded == len(DOCS) and report.n_skipped == 0
    assert report.self_recall_at_k >= 0.9
    assert report.overlap_at_k is not None and report.overlap_at_k >= 0.8  # both models retrieve the same documents
    assert not report.switched and report.blue_name is None
    assert store.load().model == "fake:a"  # the original name is still blue
    assert report.green_name in names(store)
    green = store.create_sibling(report.green_name)
    assert green.count() == len(DOCS)
    assert green.load().model == "fake:b"


def test_overlap_is_none_without_old_embedder(store):
    new = FakeEmbedder("fake:b")
    assert migrate(store, new, new_model="fake:b").overlap_at_k is None


def test_switch_makes_green_live_and_keeps_blue(store):
    original = store.name
    new = FakeEmbedder("fake:b")
    report = migrate(store, new, new_model="fake:b", switch=True)
    assert report.switched and report.status_after_switch == "same"
    # F-4: 전환 직후 판정의 수치도 보고서에 들어 있다
    assert report.verdict_after_switch is not None
    assert report.verdict_after_switch.status == "same"
    assert report.verdict_after_switch.min_cos is not None and report.verdict_after_switch.min_cos >= 0.999
    assert report.built_name == f"{original}__green" and report.green_name == original
    assert "__blue_" in report.blue_name
    assert store.name == original
    assert verify(store.load(), new, model="fake:b").status == "same"
    assert report.blue_name in names(store)
    # the new index answers queries with the new model
    top = store.query(new(["how long should dough rise for bread"])[0], 3)
    assert top and top[0] in IDS


def test_rerun_after_failure_skips_documents_already_embedded(store):
    docs_set = set(DOCS)

    class Flaky(FakeEmbedder):
        def __init__(self, fail_on_call):
            super().__init__("fake:b")
            self.fail_on_call = fail_on_call
            self.doc_texts = []

        def __call__(self, texts):
            if self.calls + 1 == self.fail_on_call:
                self.calls += 1
                raise RuntimeError("boom")
            self.doc_texts += [t[3:] for t in texts if t.startswith("d: ") and t[3:] in docs_set]
            return super().__call__(texts)

    kw = dict(new_model="fake:b", batch_size=20, doc_prefix="d: ", query_prefix="q: ")
    # call 1 = stamp probes, call 2 = first batch of 20, call 3 fails
    with pytest.raises(RuntimeError, match="boom"):
        migrate(store, Flaky(3), **kw)

    second = Flaky(0)
    report = migrate(store, second, **kw)
    assert report.n_skipped == 20 and report.n_embedded == len(DOCS) - 20
    assert len(second.doc_texts) == len(DOCS) - 20
    assert set(second.doc_texts) == set(DOCS[20:])


def test_resume_refuses_green_built_with_another_model(store):
    migrate(store, FakeEmbedder("fake:b"), new_model="fake:b")
    with pytest.raises(EmbedstampError, match="different model"):
        migrate(store, FakeEmbedder("fake:c"), new_model="fake:c")


def test_low_self_recall_refuses_to_switch(store):
    new = FakeEmbedder("fake:b")
    with pytest.raises(SwitchRefusedError, match="self-recall") as info:
        migrate(store, new, new_model="fake:b", switch=True, min_self_recall=1.01)
    assert info.value.report.switched is False
    assert store.load().model == "fake:a"


def test_empty_source_is_an_error(tmp_path):
    # F-3: 비어 있다는 사실뿐 아니라 먼저 문서를 넣으라는 대처법도 알려 준다
    with pytest.raises(EmbedstampError, match=r"empty.*add them first"):
        migrate(LocalIndex(tmp_path / "empty"), FakeEmbedder("fake:b"), new_model="fake:b")


def test_invalid_arguments(tmp_path):
    with pytest.raises(ValueError, match="batch_size"):
        migrate(make_local(tmp_path), FakeEmbedder("fake:b"), new_model="fake:b", batch_size=0)


def test_local_index_roundtrip_and_dimension_guard(tmp_path):
    idx = LocalIndex(tmp_path / "i")
    idx.add(["a", "b"], ["x", "y"], np.eye(2, 4))
    assert LocalIndex(tmp_path / "i").existing_ids() == {"a", "b"}
    with pytest.raises(ValueError, match="dimension"):
        idx.add(["c"], ["z"], np.ones((1, 5)))
    with pytest.raises(ValueError, match="unique"):
        idx.add(["d", "d"], ["z", "z"], np.ones((2, 4)))
