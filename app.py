import streamlit as st
import subprocess
import sys
import os

st.set_page_config(page_title="Ad Compare Pipeline", layout="centered")

st.title("📦 Procurement Demand Planning – Ad Compare")

# -------------------------
# Inputs
# -------------------------
bucket = st.text_input(
    "GCS Bucket Name",
    value="gc-us-gcs-aiml-dev"
)

uploaded_pdf = st.file_uploader(
    "Upload Advertisement PDF",
    type=["pdf"]
)

uploaded_excel = st.file_uploader(
    "Upload Master Excel",
    type=["xlsx"]
)

run = st.button("🚀 Run Pipeline")

# -------------------------
# Save files locally
# -------------------------
if uploaded_pdf:
    with open(uploaded_pdf.name, "wb") as f:
        f.write(uploaded_pdf.getbuffer())
    st.success("PDF uploaded successfully")

if uploaded_excel:
    with open(uploaded_excel.name, "wb") as f:
        f.write(uploaded_excel.getbuffer())
    st.success("Excel uploaded successfully")

# -------------------------
# Run pipeline via subprocess
# -------------------------
if run:
    if not uploaded_pdf or not uploaded_excel:
        st.error("Please upload both PDF and Excel files")
    else:
        try:
            with st.spinner("Running pipeline..."):
                result = subprocess.run(
                    [
                        sys.executable,
                        "main.py",
                        bucket,
                        uploaded_pdf.name,
                        uploaded_excel.name
                    ],
                    capture_output=True,
                    text=True
                )

            if result.returncode == 0:
                st.success("✅ Pipeline completed successfully")
                if result.stdout:
                    st.text(result.stdout)
            else:
                st.error("❌ Pipeline failed")
                if result.stderr:
                    st.text(result.stderr)

        except Exception as e:
            st.error(f"❌ Error: {e}")
