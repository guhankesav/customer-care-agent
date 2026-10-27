"""BM25 retrieval over the policy knowledge base (care/kb/*.md)."""
import math
import re
from collections import Counter
from pathlib import Path

KB_DIR = Path(__file__).parent / "kb"
TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN.findall(text.lower())


def load_chunks(kb_dir=KB_DIR):
    """Split each doc on '## ' headings; each section is one chunk."""
    chunks = []
    for path in sorted(kb_dir.glob("*.md")):
        title = path.stem
        for section in re.split(r"\n(?=## )", path.read_text()):
            section = section.strip()
            if section and not section.startswith("# "):
                chunks.append({"doc": title, "text": section})
    return chunks


class BM25:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks
        self.docs = [tokenize(c["text"]) for c in chunks]
        self.k1, self.b = k1, b
        self.avg = sum(map(len, self.docs)) / len(self.docs)
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def search(self, query, k=3):
        q = tokenize(query)
        scored = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    norm = f + self.k1 * (1 - self.b + self.b * len(self.docs[i]) / self.avg)
                    s += self.idf[t] * f * (self.k1 + 1) / norm
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        return [{**self.chunks[i], "score": round(s, 3)} for s, i in scored[:k]]
