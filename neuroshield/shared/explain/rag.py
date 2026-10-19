# RAG explanation layer: runs only after the gate abstains.
from __future__ import annotations
import os
import json
from dataclasses import dataclass

@dataclass
class RetrievedChunk:
    source: str
    text: str
    score: float

class LiteratureRetriever:

    def __init__(self, model_name: str='sentence-transformers/all-MiniLM-L6-v2'):
        from sentence_transformers import SentenceTransformer
        import faiss
        self.encoder = SentenceTransformer(model_name)
        self.faiss = faiss
        self.index = None
        self.chunks: list[dict] = []

    def build_index(self, docs_dir: str, chunk_chars: int=1200) -> int:
        for (root, _, files) in os.walk(docs_dir):
            for fn in sorted(files):
                if not fn.endswith(('.txt', '.md')):
                    continue
                path = os.path.join(root, fn)
                with open(path, encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                for i in range(0, len(text), chunk_chars):
                    chunk = text[i:i + chunk_chars].strip()
                    if len(chunk) > 100:
                        self.chunks.append({'source': path, 'text': chunk})
        if not self.chunks:
            raise RuntimeError(f'No documents found under {docs_dir}')
        emb = self.encoder.encode([c['text'] for c in self.chunks], normalize_embeddings=True, show_progress_bar=False)
        self.index = self.faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb.astype('float32'))
        return len(self.chunks)

    def retrieve(self, query: str, k: int=5) -> list[RetrievedChunk]:
        if self.index is None:
            raise RuntimeError('Index not built. Call build_index first.')
        q = self.encoder.encode([query], normalize_embeddings=True)
        (scores, idxs) = self.index.search(q.astype('float32'), k)
        return [RetrievedChunk(source=self.chunks[i]['source'], text=self.chunks[i]['text'], score=float(s)) for (s, i) in zip(scores[0], idxs[0])]

def build_explanation_prompt(fingerprint: dict, chunks: list[RetrievedChunk]) -> str:
    evidence = '\n\n'.join((f'[Source: {c.source}]\n{c.text}' for c in chunks))
    return f'You are explaining why a medical AI research system abstained on a case.\nUse ONLY the retrieved evidence below. If the evidence does not cover the\nfailure, say so explicitly. Do not diagnose the patient. Do not add medical\nclaims beyond the evidence. Explain for a clinician or researcher in plain\nlanguage, and cite which source each claim comes from.\n\nFAILURE FINGERPRINT (structured record of what went wrong):\n{json.dumps(fingerprint, indent=2)}\n\nRETRIEVED EVIDENCE:\n{evidence}\n\nWrite the explanation now.'

def explain_abstention(fingerprint: dict, retriever: LiteratureRetriever, llm_callable, min_score: float=0.35, k: int=5) -> dict:
    query = f"{fingerprint.get('failure_category', '')}. {fingerprint.get('description', '')}"
    chunks = retriever.retrieve(query, k=k)
    if not chunks or chunks[0].score < min_score:
        return {'explanation': 'Insufficient retrieved evidence to explain this failure. The system abstains without an evidence-grounded explanation.', 'evidence': [], 'evidence_sufficient': False}
    prompt = build_explanation_prompt(fingerprint, chunks)
    return {'explanation': llm_callable(prompt), 'evidence': [{'source': c.source, 'score': c.score} for c in chunks], 'evidence_sufficient': True}
