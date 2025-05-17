import os
import json
import io
import re
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import graphviz
import google.generativeai as genai
from pypdf import PdfReader

# -----------------------------
# Configuration & Constants
# -----------------------------
MODEL_NAME = "gemini-2.5-flash-preview-04-17"
GENERATION_PROMPT = """
You are an intelligent visual synthesis planner.
Supported visual types:
  - concept_map: expects {"concepts": {parent: [children, ...]}}
  - flowchart: expects {"steps": ["step1", "step2", ...]}
  - timeline: expects {"events": [{\"label\": str, \"date\": str}]}
  - metric_table: expects {"metrics": [{\"name\": str, \"value\": str}]}
  - fishbone: expects {"effect": str, "causes": {category: [items...]}}
Given a long text section, decide which ONE visual best represents it and output a single JSON object with keys: visual_type and data.
Output ONLY the JSON (no additional text).
"""

# -----------------------------
# Helper Functions
# -----------------------------

def load_text_from_file(upload) -> str:
    """Reads text from PDF/TXT/MD uploads."""
    if upload.type == "application/pdf":
        reader = PdfReader(upload)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text
    else:
        return upload.read().decode(errors="ignore")


def chunk_text(text: str, max_tokens: int = 2048) -> List[str]:
    """Naïve splitter for demo purposes."""
    paragraphs = text.split("\n\n")
    chunks, current, length = [], [], 0
    for p in paragraphs:
        tokens = len(p.split())
        if length + tokens > max_tokens and current:
            chunks.append("\n\n".join(current))
            current, length = [], 0
        current.append(p)
        length += tokens
    if current:
        chunks.append("\n\n".join(current))
    return chunks

# -----------------------------
# Gemini Interface with JSON extraction
# -----------------------------

def query_gemini(chunk: str) -> Dict[str, Any]:
    genai.configure(api_key="AIzaSyB5w-GjnRy-dlsCwey-Mo3OZz0uS6OulyY")
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content([GENERATION_PROMPT, chunk])
    raw = response.text or ""
    # Display raw for debugging
    st.text_area("Raw Gemini response", value=raw, height=200)
    # Extract JSON object
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        st.warning("No JSON found in Gemini response. Skipping chunk …")
        return {}
    json_str = match.group(0)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        st.warning(f"Failed to parse JSON: {e}. Skipping chunk …")
        return {}

# -----------------------------
# Visualization Generators
# -----------------------------

def generate_concept_map(data: Dict[str, List[str]]):
    g = graphviz.Digraph()
    for parent, children in data.get("concepts", {}).items():
        g.node(parent, parent)
        for child in children:
            g.node(child, child)
            g.edge(parent, child)
    return g


def generate_flowchart(data: Dict[str, List[str]]):
    g = graphviz.Digraph()
    steps = data.get("steps", [])
    for i, step in enumerate(steps):
        g.node(str(i), step)
        if i > 0:
            g.edge(str(i-1), str(i))
    return g


def generate_timeline(data: Dict[str, Any]):
    events = data.get("events", [])
    if not events:
        return None
    df = pd.DataFrame(events)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.scatter(df["date"], [1]*len(df), zorder=3)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    for _, row in df.iterrows():
        ax.text(row["date"], 1.02, row["label"], rotation=45, ha="right", va="bottom", fontsize=8)
    plt.tight_layout()
    return fig


def generate_metric_table(data: Dict[str, Any]):
    metrics = data.get("metrics", [])
    df = pd.DataFrame(metrics)
    st.table(df)
    return None


def generate_fishbone(data: Dict[str, Any]):
    g = graphviz.Digraph()
    effect = data.get("effect", "Effect")
    g.node("effect", effect, shape="box", style="filled", color="lightgrey")
    for i, (cat, causes) in enumerate(data.get("causes", {}).items()):
        cat_id = f"cat{i}"
        g.node(cat_id, cat)
        g.edge(cat_id, "effect")
        for j, c in enumerate(causes):
            cid = f"{cat_id}_{j}"
            g.node(cid, c, shape="note")
            g.edge(cid, cat_id)
    return g

VISUAL_DISPATCH = {
    "concept_map": generate_concept_map,
    "flowchart": generate_flowchart,
    "timeline": generate_timeline,
    "metric_table": generate_metric_table,
    "fishbone": generate_fishbone,
}

# -----------------------------
# Streamlit Interface
# -----------------------------

st.set_page_config(page_title="Visual Intelligence MVP", layout="wide")
st.title("📊 Visual Intelligence MVP")

upload = st.file_uploader("Upload PDF, MD or TXT", type=["pdf", "txt", "md"] , accept_multiple_files=False)

if upload is not None:
    raw_text = load_text_from_file(upload)
    st.info(f"Document loaded. Size: {len(raw_text)//1024} KB")

    with st.spinner("Analyzing & generating visuals via Gemini…"):
        chunks = chunk_text(raw_text, max_tokens=1500)
        for idx, chunk in enumerate(chunks):
            plan = query_gemini(chunk)
            v_type = plan.get("visual_type")
            data = plan.get("data")
            if not v_type or v_type not in VISUAL_DISPATCH:
                continue
            st.subheader(f"Section {idx+1}: {v_type.replace('_', ' ').title()}")
            fig_obj = VISUAL_DISPATCH[v_type](data)

            if isinstance(fig_obj, graphviz.Digraph):
                st.graphviz_chart(fig_obj)
            elif fig_obj is not None:
                st.pyplot(fig_obj)

    st.success("Done! Review the visuals above.")
else:
    st.write("⬆️ Upload a document to get started.")
