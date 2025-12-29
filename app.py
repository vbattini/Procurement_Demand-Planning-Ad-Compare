import streamlit as st
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

run = st.button("🚀 Submit Files")

# -------------------------
# Save files locally
# -------------------------
if uploaded_pdf:
    with open(uploaded_pdf.name, "wb") as f:
        f.write(uploaded_pdf.getbuffer())
    st.success("📄 PDF uploaded successfully")

if uploaded_excel:
    with open(uploaded_excel.name, "wb") as f:
        f.write(uploaded_excel.getbuffer())
    st.success("📊 Excel uploaded successfully")

# -------------------------
# UI-ONLY action
# -------------------------
if run:
    if not uploaded_pdf or not uploaded_excel:
        st.error("❌ Please upload both PDF and Excel files")
    else:
        st.success("✅ Files received successfully")
        st.info(
            """
            🔧 **Pipeline Execution Notice**

            The ML pipeline runs in a separate backend environment
            (Local / VM / GCP / Colab).

            This Streamlit app is used for:
            • Uploading inputs  
            • Validating files  
            • Triggering backend jobs (future)  
            • Viewing results  

            Please run the pipeline using:
            ```
            python main.py <bucket> <pdf> <excel>
            ```
            """
        )
