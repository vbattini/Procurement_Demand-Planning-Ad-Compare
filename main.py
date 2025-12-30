import time
import sys

from objectdetection import PDFProcessingPipeline
from textextraction import TextExtractionPipeline
from textcomparison_nlp import NLPMatchingComponent
from textcomparison_llm import LLMSimilarityComponent
from textcomparison_finalresults import FinalMatchComponent


def run_pipeline(bucket: str, input_pdf: str, master_excel: str):
    print("🚀 Starting Ad Compare Pipeline\n")

    print(f"📦 GCS Bucket   : {bucket}")
    print(f"📄 Input PDF   : {input_pdf}")
    print(f"📊 Master XLSX : {master_excel}\n")

    # ----------------------------
    # Step 1: Object Detection
    # ----------------------------
    print("🔹 Step 1: Object Detection Pipeline")
    t1 = time.time()

    pdf_pipeline = PDFProcessingPipeline(bucket, input_pdf)
    pdf_pipeline.run()

    print(f"✅ Object Detection completed in {time.time() - t1:.2f}s\n")

    # ----------------------------
    # Step 2: Text Extraction
    # ----------------------------
    print("🔹 Step 2: Text Extraction Pipeline")
    t2 = time.time()

    extraction_pipeline = TextExtractionPipeline(bucket, input_pdf)
    extraction_pipeline.run()

    print(f"✅ Text Extraction completed in {time.time() - t2:.2f}s\n")

    # ----------------------------
    # Step 3.1: NLP Comparison
    # ----------------------------
    print("🔹 Step 3.1: NLP Matching Pipeline")
    t3 = time.time()

    nlp_pipeline = NLPMatchingComponent(bucket, input_pdf, master_excel)
    nlp_pipeline.run()

    print(f"✅ NLP Matching completed in {time.time() - t3:.2f}s\n")

    # ----------------------------
    # Step 3.2: LLM Similarity
    # ----------------------------
    print("🔹 Step 3.2: LLM Similarity Pipeline")
    t4 = time.time()

    llm_pipeline = LLMSimilarityComponent(bucket, input_pdf, master_excel)
    llm_pipeline.run()

    print(f"✅ LLM Similarity completed in {time.time() - t4:.2f}s\n")

    # ----------------------------
    # Step 3.3: Final Results
    # ----------------------------
    print("🔹 Step 3.3: Final Result Generation")
    t5 = time.time()

    final_pipeline = FinalMatchComponent(bucket, input_pdf, master_excel)
    final_pipeline.run()

    print(f"✅ Final Results generated in {time.time() - t5:.2f}s\n")

    print("🎉 Pipeline completed successfully")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "❌ Usage:\n"
            "python main.py <bucket> <input_pdf> <master_excel>\n\n"
            "Example:\n"
            "python main.py gc-us-gcs-aiml-dev inputs/NEW_PW.pdf inputs/master.xlsx"
        )
        sys.exit(1)

    _, bucket, input_pdf, master_excel = sys.argv
    run_pipeline(bucket, input_pdf, master_excel)
