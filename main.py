!python -m pip install --upgrade pip
import os
os.system("pip install opencv-python-headless")
from objectdetection import PDFProcessingPipeline
from textextraction  import TextExtractionPipeline
from textcomparison_nlp  import NLPMatchingComponent 
from textcomparison_llm import LLMSimilarityComponent
from textcomparison_finalresults import FinalMatchComponent

import time



def main():
    # ----------------------------
    # Configuration
    # ----------------------------
    bucket = "gc-us-gcs-aiml-dev"          # GCS bucket name
    input_pdf = "NEW_PW.pdf"  
    master_excel = "PWMW B&W 9.9.25 Ad Test.xlsx"      # Input PDF file name in GCS

    # ----------------------------
    # Step 1: Run Object Detection Pipeline
    # ----------------------------
    print("🔹 Running Object Detection Pipeline...")
    start_time = time.time()

    pdf_pipeline = PDFProcessingPipeline(bucket, input_pdf,verbose=False)
    pdf_pipeline.run()

    print(f"⏱ Object Detection Pipeline completed in {time.time() - start_time:.2f}s\n")

    # step 2:Text Extraction pipeline
    print("🔹 Running Text Extraction Pipeline...")
    step2_start = time.time()

    extraction_pipeline = TextExtractionPipeline(bucket, input_pdf)
    extraction_pipeline.run()

    step2_time = time.time() - step2_start
    print(f"✅ Text Extraction completed in {step2_time:.2f} seconds.\n")


    # step 3:Text comparison pipeline
    ## step 3.1 - NLP comparison

    print("🔹 Running NLP Comparison Pipeline...")
    step3_start = time.time()

    nlp_pipeline = NLPMatchingComponent(bucket,input_pdf, master_excel)
    nlp_pipeline.run()

    step3_time = time.time() - step3_start
    print(f"✅ NLP Comparison completed in {step3_time:.2f} seconds.\n")

    ## step 3.2 - Finetuned LLM comparison
    print("🔹 Running LLM Similarity Scoring Pipeline...")
    step4_start = time.time()
    llm_component = LLMSimilarityComponent(bucket, input_pdf,master_excel)
    llm_component.run()
    step4_time = time.time() - step4_start

    print(f"✅ LLM similarity scores generations completed in {step4_time:.2f} seconds.\n")

    ## step 3.3 - Feature level Match + Final Results
    print("🔹 Generating Final Results...")
    step5_start = time.time()
    finalresult_component = FinalMatchComponent(bucket, input_pdf,master_excel)
    finalresult_component.run()
    step5_time = time.time() - step5_start

    print(f"✅ Final results generations completed in {step5_time:.2f} seconds.\n")






if __name__ == "__main__":
    main()

