"""Chroma + Guard: stamp a collection and refuse queries from a swapped embedding model.

    uv run python examples/chroma_guard.py --mock     # practice mock, offline
    uv run python examples/chroma_guard.py            # real fastembed models (downloads ~167 MB once)

Real mode needs:  pip install "embedstamp[chroma,fastembed]"
"""

import argparse
import tempfile

import chromadb
from chromadb.config import Settings

from embedstamp import EmbeddingMismatchError, Guard, resolve_embedder, stamp
from embedstamp.stores.chroma import ChromaStore

parser = argparse.ArgumentParser()
parser.add_argument("--mock", action="store_true", help="use fake embedders (no download)")
args = parser.parse_args()

if args.mock:
    spec_a, spec_b = "fake:a", "fake:b"
else:
    spec_a, spec_b = "fastembed:BAAI/bge-small-en-v1.5", "fastembed:sentence-transformers/all-MiniLM-L6-v2"

embed_a, model_a = resolve_embedder(spec_a)
embed_b, model_b = resolve_embedder(spec_b)
docs = ["Reset your password in settings.", "Invoices are due in thirty days.", "Plants make sugar from light."]

with tempfile.TemporaryDirectory() as tmp:
    client = chromadb.PersistentClient(path=tmp, settings=Settings(anonymized_telemetry=False))
    # 벡터를 직접 넣는다(embedding_function=None): Chroma 자체 검사는 여기서 아무것도 하지 않는다
    collection = client.get_or_create_collection("docs", embedding_function=None)
    collection.add(ids=["1", "2", "3"], documents=docs, embeddings=embed_a(docs).tolist())

    store = ChromaStore(collection)
    store.save(stamp(embed_a, model=model_a))  # 컬렉션 메타데이터 'embedstamp' 키에 저장

    ok = Guard(store, embed_a, model=model_a)
    print("same model  :", ok.check().status)

    bad = Guard(store, embed_b, model=model_b)
    try:
        bad.embed_query("forgot my login")
    except EmbeddingMismatchError as exc:
        print("swapped model: EmbeddingMismatchError caught ->", exc.verdict.status, exc.verdict.reasons)
