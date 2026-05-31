
import glob
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from config import Config
from kestrel_rag import KestrelEngine
#import load_json
import gradio as gr



import textwrap
import plotly.graph_objects as go


# Color scheme by source type
COLORS = {
    "kev": "#1D9E75",   # teal   — CISA KEV entries
    "nvd": "#185FA5",   # blue   — NVD records
    "pdf": "#5F5E5A",   # gray   — PDF documents
}


def _source_type(chunk: dict) -> str:
    """Determine source type for color coding."""
    source = chunk.get("source", "") or chunk.get("metadata", {}).get("source", "")
    if source.endswith(".pdf"):
        return "pdf"
    if chunk.get("metadata", {}).get("in_kev", False):
        return "kev"
    return "nvd"


def _wrap_text(text: str, width: int = 80, max_lines: int = 10) -> str:
    """
    Wrap chunk text for tooltip display.
    Plotly hover text renders <br> as line breaks.
    """
    lines  = []
    for paragraph in text.split("\n"):
        wrapped = textwrap.wrap(paragraph, width=width)
        lines.extend(wrapped if wrapped else [""])

    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["..."]

    return "<br>".join(lines)


def plot_retrieval_distances(
    chunks:   list[dict],
    query:    str = "",
    top_k:    int = None,
) -> go.Figure:
    """
    Build an interactive Plotly bar chart of retrieval similarity scores.

    Each bar represents one retrieved chunk. Hovering shows:
        - Chunk ID / source
        - Similarity score
        - Source type (KEV / NVD / PDF)
        - Full chunk text preview (wrapped)

    Clicking a bar in a Gradio context selects it for inspection.

    Args:
        chunks:  List of chunk dicts from retrieve() or query()["chunks"].
                 Each dict must have "text", "source" or metadata["source"],
                 and optionally "score" / "distance".
        query:   The original query string — shown in the chart title.
        top_k:   If provided, limits display to top_k chunks.

    Returns:
        go.Figure: Interactive Plotly figure ready for .show() or gr.Plot().
    """
    
    if top_k:
        chunks = chunks[:top_k]

    if not chunks:
        fig = go.Figure()
        fig.update_layout(title="No chunks retrieved")
        return fig

   
    # Build per-bar data
   
    chunk_ids   = []
    scores      = []
    hover_texts = []
    colors      = []
    labels      = []

    for i, chunk in enumerate(chunks):
        
        # ID / label
        cid   = chunk.get("id", f"chunk_{i}")
        label = f"chunk {i + 1}"
        chunk_ids.append(cid)
        labels.append(label)

        # Score — ChromaDB returns distance (lower = closer); convert to similarity
        raw = chunk.get("score") or chunk.get("distance")
        if raw is None:
            score = 0.0
        elif raw > 1.0:
            # Raw L2 distance — normalize to 0-1 similarity
            score = round(1 / (1 + raw), 3)
        else:
            # Already a similarity score (cosine)
            score = round(float(raw), 3)
        
        scores.append(score)

        # Source type and color
        stype = _source_type(chunk)
        colors.append(COLORS[stype])

        # Source filename
        source = (
            chunk.get("source")
            or chunk.get("metadata", {}).get("source", "unknown")
        )

        # KEV flag
        in_kev   = chunk.get("metadata", {}).get("in_kev", False)
        kev_line = "<b>⚑ CISA KEV</b><br>" if in_kev else ""

        # Chunk text preview
        text_preview = _wrap_text(chunk.get("text", ""), width=80, max_lines=12)

        hover = (
            f"<b>{cid}</b><br>"
            f"source: {source}<br>"
            f"type: {stype.upper()}<br>"
            f"score: {score:.3f}<br>"
            f"{kev_line}"
            f"<br>"
            f"<span style='color:#888'>— chunk preview —</span><br>"
            f"<span style='font-family:monospace;font-size:11px'>{text_preview}</span>"
        )
        hover_texts.append(hover)

   
    # Build figure
   
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=labels,
        y=scores,
        marker_color=colors,
        marker_line_color=[c.replace("75", "56") for c in colors],
        marker_line_width=1.5,
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_texts,
        text=[f"{s:.3f}" for s in scores],
        textposition="outside",
        textfont=dict(size=11),
    ))

    # Threshold line at 0.5 — chunks below this are weak matches
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="rgba(150,150,150,0.5)",
        annotation_text="0.5 threshold",
        annotation_position="right",
        annotation_font_size=11,
    )

   
    # Layout
   
    title_text = (
        f"Retrieval scores — top {len(chunks)} chunks"
        + (f"<br><sup>query: \"{query[:80]}{'...' if len(query) > 80 else ''}\"</sup>"
           if query else "")
    )

    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=14),
            x=0,
        ),
        xaxis=dict(
            title="retrieved chunks",
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            title="cosine similarity",
            range=[0, 1.1],
            tickfont=dict(size=11),
            gridcolor="rgba(150,150,150,0.15)",
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="monospace",
            bordercolor="#cccccc",
            namelength=-1,
        ),
        margin=dict(t=80, b=60, l=60, r=40),
        height=380,
        showlegend=False,
        bargap=0.35,
    )

   
    # Manual color legend as annotations
   
    legend_items = [
        ("KEV entry",    COLORS["kev"]),
        ("NVD record",   COLORS["nvd"]),
        ("PDF document", COLORS["pdf"]),
    ]
    for i, (label, color) in enumerate(legend_items):
        fig.add_annotation(
            x=1.0,
            y=1.0 - (i * 0.08),
            xref="paper",
            yref="paper",
            text=f"<span style='color:{color}'>■</span> {label}",
            showarrow=False,
            xanchor="right",
            font=dict(size=11),
        )

    return fig


import gradio as gr


def build_ui(kr: KestrelEngine) -> gr.Blocks:

    top_k = kr.config.top_k

    def chat(user_message: str, history: list[dict]):
        chunks  = kr.retrieve(user_message, top_k)
        result  = kr.query(user_message, chunks)
        sources = sorted({c["source"] for c in chunks})
        answer  = f"{result['answer']}\n\n*Sources: {sources}*"
        fig     = plot_retrieval_distances(chunks, user_message)
        return answer, fig

    with gr.Blocks(title="Kestrel — CVE/KEV Assistant") as ui:

        gr.Markdown("## Kestrel — CVE / KEV Vulnerability Assistant")

        with gr.Row():

            # --- Left column: chat ---
            with gr.Column(scale=2):
                chat_ui = gr.ChatInterface(
                    fn=chat,
                    chatbot=gr.Chatbot(height=460),
                    autofocus=True,
                    additional_outputs=[gr.Plot(label="Retrieval distances")],
                )

            

    return ui  
    
def main():
    
    # Load config and print settings
    config = Config()
    
    if config.client is None:
        print("Error: OpenAI client not initialized. Please set OPENROUTER_API_KEY in your environment.")
        return
    else:
        print("OpenAI client initialized successfully!")
        kestrel = KestrelEngine(config)
        print("KestrelEngine initialized successfully!")
        # Load data and build index
        print("Loading data and building index...")
        kestrel.load()
        print ("Chunking and indexing data...")
        kestrel.chunk()
        (f"Indexing data into ChromaDB at '{kestrel.config.chroma_dir}'...")
        kestrel.index()
        print("Indexing complete!")
        if kestrel.collection:
            print(f"Collection '{kestrel.config.collection_name}' now contains {kestrel.collection.count()} chunks.")
            ui = build_ui(kestrel)
            ui.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)  # launch Gradio UI on all interfaces at port 7860

if __name__ == "__main__":
    main()

