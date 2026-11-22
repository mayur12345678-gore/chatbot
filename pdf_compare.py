"""
Multi-PDF Comparison Tool
=========================
Upload 2-3 PDFs and ask comparison questions across them.

Pipeline
--------
1. Each PDF gets its OWN retriever (its own chunks + its own index), so one
   long document can never crowd out the others.
2. For every question, retrieve the top-k chunks from EACH PDF.
3. ONE prompt contains all retrieved context (labelled per document) and asks
   the LLM for a comparison table.

Run the app:   streamlit run pdf_compare.py
Use as a lib:  from pdf_compare import compare_pdfs
"""
from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
MIN_PDFS, MAX_PDFS = 2, 3


# --------------------------------------------------------------------------
# 1. Loading and chunking
# --------------------------------------------------------------------------
@dataclass
class Chunk:
    doc: str          # PDF file name
    page: int         # 1-based page number
    text: str
    score: float = 0.0


def load_pdf_pages(source) -> List[str]:
    """Return the text of each page. `source` is a path or a file-like object."""
    reader = PdfReader(source)
    return [(page.extract_text() or "").strip() for page in reader.pages]


def chunk_pages(doc: str, pages: Sequence[str],
                size: int = 900, overlap: int = 150) -> List[Chunk]:
    """Split each page into overlapping character windows, keeping page numbers."""
    chunks: List[Chunk] = []
    step = max(size - overlap, 1)
    for page_no, text in enumerate(pages, start=1):
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        for start in range(0, len(text), step):
            piece = text[start:start + size].strip()
            if len(piece) > 40:                      # skip tiny tail fragments
                chunks.append(Chunk(doc=doc, page=page_no, text=piece))
            if start + size >= len(text):
                break
    return chunks


# --------------------------------------------------------------------------
# 2. One retriever per PDF
# --------------------------------------------------------------------------
class PDFRetriever:
    """
    Independent index for a single PDF.

    Uses TF-IDF so it runs offline with no model download. To use dense
    embeddings instead, keep the same two methods (`__init__`, `retrieve`)
    and swap the internals (e.g. sentence-transformers + FAISS).
    """

    def __init__(self, name: str, pages: Sequence[str]):
        self.name = name
        self.chunks = chunk_pages(name, pages)
        if not self.chunks:
            raise ValueError(f"No extractable text in '{name}' (scanned PDF? try OCR).")
        # Fitted per document, so IDF reflects THIS document only.
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), sublinear_tf=True
        )
        self.matrix = self.vectorizer.fit_transform(c.text for c in self.chunks)

    def retrieve(self, query: str, k: int = 4) -> List[Chunk]:
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        top = scores.argsort()[::-1][:k]
        return [
            Chunk(self.chunks[i].doc, self.chunks[i].page, self.chunks[i].text, float(scores[i]))
            for i in top
            if scores[i] > 0
        ]


def build_retrievers(files: Dict[str, object]) -> Dict[str, PDFRetriever]:
    """files: {display_name: path_or_file_object} -> {display_name: PDFRetriever}"""
    if not (MIN_PDFS <= len(files) <= MAX_PDFS):
        raise ValueError(f"Please provide {MIN_PDFS}-{MAX_PDFS} PDFs (got {len(files)}).")
    return {name: PDFRetriever(name, load_pdf_pages(src)) for name, src in files.items()}


# --------------------------------------------------------------------------
# 3. Retrieve from each, then ONE comparison prompt
# --------------------------------------------------------------------------
def retrieve_all(retrievers: Dict[str, PDFRetriever], question: str,
                 k: int = 4) -> Dict[str, List[Chunk]]:
    return {name: r.retrieve(question, k) for name, r in retrievers.items()}


SYSTEM_PROMPT = (
    "You are a careful document analyst. Answer ONLY from the provided excerpts. "
    "Never use outside knowledge. If an excerpt does not cover something, say so."
)


def build_prompt(question: str, retrieved: Dict[str, List[Chunk]]) -> str:
    blocks = []
    for label, (name, chunks) in zip("ABC", retrieved.items()):
        body = "\n".join(f"[p.{c.page}] {c.text}" for c in chunks) or "(no relevant passages found)"
        blocks.append(f"=== DOCUMENT {label}: {name} ===\n{body}")
    context = "\n\n".join(blocks)
    columns = " | ".join(f"Document {l} ({n})" for l, n in zip("ABC", retrieved))
    return f"""Compare the documents below to answer the question.

{context}

QUESTION: {question}

INSTRUCTIONS
1. Output a Markdown comparison table. First column "Aspect"; then one column per document: {columns}.
2. Choose rows that matter for the question (for example: key figures, terms, dates, conditions, scope).
3. Cite pages in each cell like (p.3). If a document does not mention an aspect, write "Not found in document".
4. After the table, write "Key differences" (2-4 bullets) and, if relevant, "Key similarities" (1-2 bullets).
5. Do not invent facts that are not in the excerpts."""


def call_claude(prompt: str, api_key: str | None = None, model: str = MODEL) -> str:
    import anthropic  # pip install anthropic

    client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
    msg = client.messages.create(
        model=model,
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")


def compare_pdfs(retrievers: Dict[str, PDFRetriever], question: str, k: int = 4,
                 llm: Callable[[str], str] = call_claude):
    """Full pipeline. Returns (answer_markdown, retrieved_chunks, prompt)."""
    retrieved = retrieve_all(retrievers, question, k)
    prompt = build_prompt(question, retrieved)
    return llm(prompt), retrieved, prompt


# --------------------------------------------------------------------------
# 4. Report export (PDF)
# --------------------------------------------------------------------------
def _parse_md_table(lines: List[str]) -> List[List[str]]:
    rows = []
    for ln in lines:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):   # separator row
            continue
        rows.append(cells)
    return rows


def _inline(md: str) -> str:
    """Tiny markdown -> reportlab markup (escape, then **bold**)."""
    md = md.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", md)


def export_report_pdf(question: str, answer_md: str,
                      retrieved: Dict[str, List[Chunk]]) -> bytes:
    from datetime import date
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer, Table,
                                    TableStyle, KeepTogether)

    ss = getSampleStyleSheet()
    cell = ParagraphStyle("cell", parent=ss["BodyText"], fontSize=8.5, leading=11, alignment=TA_LEFT)
    head = ParagraphStyle("head", parent=cell, textColor=colors.white, fontName="Helvetica-Bold")
    small = ParagraphStyle("small", parent=cell, fontSize=8, textColor=colors.HexColor("#555555"))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=15 * mm, bottomMargin=15 * mm,
                            title="PDF Comparison Report")
    story = [
        Paragraph("PDF Comparison Report", ss["Title"]),
        Paragraph(f"{date.today():%d %B %Y} &nbsp;|&nbsp; Documents: "
                  + ", ".join(_inline(n) for n in retrieved), small),
        Spacer(1, 6 * mm),
        Paragraph("Question", ss["Heading2"]),
        Paragraph(_inline(question), ss["BodyText"]),
        Spacer(1, 4 * mm),
        Paragraph("Comparison", ss["Heading2"]),
    ]

    # Split the answer into table lines and prose lines.
    table_lines, prose = [], []
    for ln in answer_md.splitlines():
        (table_lines if ln.strip().startswith("|") else prose).append(ln)
    rows = _parse_md_table(table_lines)
    if rows:
        ncol = len(rows[0])
        data = [[Paragraph(_inline(c), head) for c in rows[0]]]
        for r in rows[1:]:
            r = (r + [""] * ncol)[:ncol]
            data.append([Paragraph(_inline(c), cell) for c in r])
        width = landscape(A4)[0] - 30 * mm
        first = 35 * mm
        t = Table(data, colWidths=[first] + [(width - first) / (ncol - 1)] * (ncol - 1), repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3A5F")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F5F9")]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C5CCD6")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ]))
        story.append(t)
    for ln in prose:
        ln = ln.strip()
        if not ln:
            story.append(Spacer(1, 2 * mm))
        elif ln.startswith(("-", "*")):
            story.append(Paragraph("&bull; " + _inline(ln.lstrip("-* ").strip()), ss["BodyText"]))
        else:
            story.append(Paragraph(_inline(ln.lstrip("# ").strip()), ss["BodyText"]))

    story += [Spacer(1, 6 * mm), Paragraph("Retrieved sources", ss["Heading2"])]
    for name, chunks in retrieved.items():
        story.append(Paragraph(f"<b>{_inline(name)}</b>", ss["BodyText"]))
        for c in chunks:
            story.append(KeepTogether([Paragraph(
                f"p.{c.page} (score {c.score:.2f}): {_inline(c.text[:350])}...", small)]))
        story.append(Spacer(1, 3 * mm))
    doc.build(story)
    return buf.getvalue()


# --------------------------------------------------------------------------
# 5. Streamlit UI
# --------------------------------------------------------------------------
def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Multi-PDF Comparison Tool", layout="wide")
    st.title("Multi-PDF Comparison Tool")
    st.caption("Upload 2-3 PDFs, then ask comparison questions across them.")

    with st.sidebar:
        st.header("Setup")
        api_key = st.text_input("Anthropic API key", type="password",
                                value=os.getenv("ANTHROPIC_API_KEY", ""))
        uploads = st.file_uploader("PDFs (2-3)", type="pdf", accept_multiple_files=True)
        k = st.slider("Passages retrieved per PDF", 2, 8, 4)

    if len(uploads) > MAX_PDFS:
        st.error(f"Maximum {MAX_PDFS} PDFs. Remove {len(uploads) - MAX_PDFS}.")
        st.stop()
    if len(uploads) < MIN_PDFS:
        st.info(f"Upload at least {MIN_PDFS} PDFs to begin.")
        st.stop()

    # One retriever per PDF, rebuilt only when the uploaded set changes.
    sig = tuple((u.name, u.size) for u in uploads)
    if st.session_state.get("sig") != sig:
        with st.spinner("Indexing each PDF separately..."):
            try:
                st.session_state.retrievers = build_retrievers(
                    {u.name: io.BytesIO(u.getvalue()) for u in uploads})
            except ValueError as e:
                st.error(str(e))
                st.stop()
        st.session_state.sig = sig
        st.session_state.history = []
    retrievers = st.session_state.retrievers

    cols = st.columns(len(retrievers))
    for col, (name, r) in zip(cols, retrievers.items()):
        col.metric(name, f"{len(r.chunks)} chunks")

    question = st.chat_input("Ask a comparison question, e.g. 'How do the refund policies differ?'")
    if question:
        if not api_key:
            st.warning("Enter your Anthropic API key in the sidebar.")
            st.stop()
        with st.spinner("Retrieving from each PDF and comparing..."):
            answer, retrieved, _ = compare_pdfs(
                retrievers, question, k, llm=lambda p: call_claude(p, api_key))
        st.session_state.history.append((question, answer, retrieved))

    for q, answer, retrieved in reversed(st.session_state.history):
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.markdown(answer)
            with st.expander("Retrieved passages"):
                for name, chunks in retrieved.items():
                    st.markdown(f"**{name}**")
                    for c in chunks:
                        st.caption(f"p.{c.page} - score {c.score:.2f}")
                        st.write(c.text)
            st.download_button(
                "Download report (PDF)",
                export_report_pdf(q, answer, retrieved),
                file_name="comparison_report.pdf",
                mime="application/pdf",
                key=f"dl_{abs(hash(q + answer))}",
            )


if __name__ == "__main__":
    main()
