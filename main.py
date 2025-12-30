from objectdetection import PDFProcessingPipeline
from textextraction import TextExtractionPipeline
from textcomparison_nlp import NLPMatchingComponent
from textcomparison_llm import LLMSimilarityComponent
from textcomparison_finalresults import FinalMatchComponent
import time


def run_pipeline(bucket, input_pdf, master_excel):
    # ----------------------------
    # Step 1: Object Detection
    # ----------------------------
    print("🔹 Running Object Detection Pipeline...")
    start_time = time.time()

    pdf_pipeline = PDFProcessingPipeline(bucket, input_pdf, verbose=False)
    pdf_pipeline.run()

    print(f"⏱ Object Detection completed in {time.time() - start_time:.2f}s\n")

    # ----------------------------
    # Step 2: Text Extraction
    # ----------------------------
    print("🔹 Running Text Extraction Pipeline...")
    step2_start = time.time()

    extraction_pipeline = TextExtractionPipeline(bucket, input_pdf)
    extraction_pipeline.run()

    print(f"✅ Text Extraction completed in {time.time() - step2_start:.2f}s\n")

    # ----------------------------
    # Step 3.1: NLP Comparison
    # ----------------------------
    print("🔹 Running NLP Comparison Pipeline...")
    step3_start = time.time()

    nlp_pipeline = NLPMatchingComponent(bucket, input_pdf, master_excel)
    nlp_pipeline.run()

    print(f"✅ NLP Comparison completed in {time.time() - step3_start:.2f}s\n")

    # ----------------------------
    # Step 3.2: LLM Similarity
    # ----------------------------
    print("🔹 Running LLM Similarity Scoring Pipeline...")
    step4_start = time.time()

    llm_component = LLMSimilarityComponent(bucket, input_pdf, master_excel)
    llm_component.run()

    print(f"✅ LLM similarity completed in {time.time() - step4_start:.2f}s\n")

    # ----------------------------
    # Step 3.3: Final Results
    # ----------------------------
    print("🔹 Generating Final Results...")
    step5_start = time.time()

    finalresult_component = FinalMatchComponent(bucket, input_pdf, master_excel)
    finalresult_component.run()

    print(f"✅ Final results completed in {time.time() - step5_start:.2f}s\n")

    print("🎉 PIPELINE COMPLETED SUCCESSFULLY")


# ✅ Keeps CLI execution working
if __name__ == "__main__":
    run_pipeline(
        bucket="gc-us-gcs-aiml-dev",
        input_pdf="NEW_PW.pdf",
        master_excel="PWMW B&W 9.9.25 Ad Test.xlsx"
    )
