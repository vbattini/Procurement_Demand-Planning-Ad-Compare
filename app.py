import sys
import streamlit as st
st.write(sys.version)

import streamlit as st
from main import run_pipeline

st.set_page_config(page_title="Ad Compare Pipeline", layout="centered")
st.title("📦 Procurement Demand Planning – Ad Compare")

bucket = st.text_input("GCS Bucket Name", value="gc-us-gcs-aiml-dev")

uploaded_pdf = st.file_uploader("Upload Advertisement PDF", type=["pdf"])
uploaded_excel = st.file_uploader("Upload Master Excel", type=["xlsx"])

run = st.button("🚀 Run Pipeline")

if uploaded_pdf:
    with open(uploaded_pdf.name, "wb") as f:
        f.write(uploaded_pdf.getbuffer())
    st.success("PDF uploaded successfully")

if uploaded_excel:
    with open(uploaded_excel.name, "wb") as f:
        f.write(uploaded_excel.getbuffer())
    st.success("Excel uploaded successfully")

if run:
    if not uploaded_pdf or not uploaded_excel:
        st.error("Please upload both PDF and Excel files")
    else:
        with st.spinner("Running pipeline..."):
            run_pipeline(
                bucket=bucket,
                input_pdf=uploaded_pdf.name,
                master_excel=uploaded_excel.name
            )
        st.success("✅ Pipeline completed successfully")
