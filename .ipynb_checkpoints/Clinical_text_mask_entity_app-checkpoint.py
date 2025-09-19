# Clinical_mask_entity_App.py

import streamlit as st
import pandas as pd
from html import escape
import re
from io import StringIO
import subprocess
import sys

# ---------------------------
# Streamlit App Config
# ---------------------------
st.set_page_config(
    page_title="Clinical Notes De-ID & Entity Extraction",
    layout="wide"
)
st.title("Clinical Notes De-Identification & Entity Extraction")

# ---------------------------
# Ensure spaCy model is installed
# ---------------------------
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    import spacy
    nlp = spacy.load("en_core_web_sm")

# ---------------------------
# Import your pipelines
# ---------------------------
# Make sure these files are in the repo root or subfolder and included in GitHub
from deid_pipeline_fixed import deidentify_text
from entity_extract_pipeline import extract_entities_from_text

# ---------------------------
# File Upload
# ---------------------------
uploaded_file = st.file_uploader("Upload a clinical notes .txt file", type=["txt"])

if uploaded_file:
    text = uploaded_file.read().decode("utf-8")
    masked_text = deidentify_text(text)
    df_entities = extract_entities_from_text(masked_text)

    # -------------------------
    # Toggle between masked and original
    # -------------------------
    view_option = st.radio("View Text:", ("Masked / De-identified", "Original"))
    st.subheader("Clinical Text")
    display_text = masked_text if view_option == "Masked / De-identified" else text

    # -------------------------
    # Highlight entities in masked text
    # -------------------------
    if view_option == "Masked / De-identified":
        colors = {
            "DISEASE": "lightcoral",
            "SYMPTOM": "orange",
            "MEDICATION": "lightgreen",
            "LAB_TEST": "lightblue",
            "PROCEDURE": "violet"
        }

        highlighted_text = escape(display_text)
        for _, row in df_entities.iterrows():
            entity_html = f'<span style="background-color:{colors.get(row["Type"], "yellow")};">{escape(row["Entity"])}</span>'
            highlighted_text = highlighted_text.replace(row["Entity"], entity_html)

        st.markdown(
            f"<div style='font-family:monospace; white-space:pre-wrap'>{highlighted_text}</div>",
            unsafe_allow_html=True
        )
    else:
        st.text(display_text)

    # -------------------------
    # PHI Metrics
    # -------------------------
    phi_patterns = {
        "Names": r"\[NAME\]",
        "Addresses": r"\[ADDRESS\]",
        "Contacts": r"\[CONTACT\]",
        "IDs": r"\[ID\]",
        "Dates": r"\[DATE\]"
    }
    phi_counts = {k: len(re.findall(v, masked_text)) for k, v in phi_patterns.items()}
    total_phi = sum(phi_counts.values())

    st.subheader("PHI & Entity Metrics")
    st.write(f"**Total PHI masked:** {total_phi}")
    st.write("**PHI Breakdown:**")
    st.json(phi_counts)
    st.write(f"**Total entities extracted:** {len(df_entities)}")

    # -------------------------
    # Entity counts with names
    # -------------------------
    if not df_entities.empty:
        entity_counts = df_entities.groupby(['Entity', 'Type']).size().reset_index(name='Count')
        st.subheader("Entities Extracted (with Counts)")
        st.dataframe(entity_counts)

        # Bar chart of entity counts by type
        chart_data = df_entities['Type'].value_counts()
        st.subheader("Entity Types Distribution")
        st.bar_chart(chart_data)

        # -------------------------
        # Download buttons
        # -------------------------
        st.download_button("Download Masked Text", masked_text, "masked_clinical_notes.txt", "text/plain")

        csv_buffer = StringIO()
        entity_counts.to_csv(csv_buffer, index=False)
        st.download_button("Download Entity Table CSV", csv_buffer.getvalue(), "extracted_entities.csv", "text/csv")
    else:
        st.info("No entities were extracted from the text.")
