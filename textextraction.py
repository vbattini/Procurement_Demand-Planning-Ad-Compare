# text_extraction.py
import os
import re
import json
import time
import sys
import base64
import random
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import storage
import google.genai as genai

from IPython.display import HTML, Markdown, display
from google import genai
from google.genai.types import (
    FunctionDeclaration,
    GenerateContentConfig,
    GoogleSearch,
    HarmBlockThreshold,
    HarmCategory,
    MediaResolution,
    Part,
    Retrieval,
    SafetySetting,
    Tool,
    ToolCodeExecution,
    VertexAISearch,
)




if "google.colab" in sys.modules:
    from google.colab import auth

    auth.authenticate_user()



PROJECT_ID = "gc-proj-aiml-dev-01fd"  # @param {type: "string", placeholder: "[your-project-id]", isTemplate: true}
if not PROJECT_ID or PROJECT_ID == "[your-project-id]":
    PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))

LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# ====================== TEXT EXTRACTION PIPELINE ======================

class TextExtractionPipeline:
    def __init__(self, gcs_bucket_name: str, input_pdf: str, root_prefix: str = "DPAC/6.pipelineRun"):
        # Base inputs
        self.GCS_BUCKET_NAME = gcs_bucket_name
        self.INPUT_PDF = input_pdf
        self.GCS_PIPELINE_ROOT_PREFIX = root_prefix

        # Derived paths
        self.pdf_name_only = os.path.splitext(os.path.basename(self.INPUT_PDF))[0]
        self.PDF_ROOT_PREFIX = f"{self.GCS_PIPELINE_ROOT_PREFIX}/{self.pdf_name_only}"
        self.CLEANED_IMAGES_PREFIX = f"{self.PDF_ROOT_PREFIX}/cleaned_images_new"
        self.OUTPUT_FILE = "final_extracted_text.xlsx"
        self.DESTINATION_BLOB_NAME = f"{self.PDF_ROOT_PREFIX}/Text_Extraction.xlsx"

        # Concurrency and retry configuration
        self.THREAD_COUNT = 100
        self.RETRY_LIMIT = 3

        # Logging
        self.START_TIME = datetime.now()
        global client
        self.client = client


        # Tracking lists
        self.classification_failures = []
        self.classification_success = []
        self.extraction_failures = []
        self.extraction_success = []

        # Prompts (kept private)
        self.CLASSIFICATION_PROMPT = """ Classify the given image into one of the following clusters based on text placement and product details.

        Below are some examples of cluster 1 images:
        =======
        Cluster 1:
        - The text is located on the right side or top-right side of the image and have one product decription.



          example 1:
          Input text:'2/$4 18-19 oz Select Old El Paso or Progresso Soup'

          example 2:
          Input Text: '5.99lb 90% Lean Ground Beef'


        ======


        Cluster 2 images contain details of multiple products with associated size measurement. Below are some examples of cluster 2 images:
        ======
        Cluster 2:

        - The image contains multiple product Descriptons.
        - The text description is long and covers a larger portion of the image.



          example 1:
          Input text:'4/$5 7-16 oz Barilla Pasta 16-24 oz Ragu Pasta Sauce 1'

          example 2:
          Input text:'4.99 10 oz Raybern's Sandwiches or 13-33 oz Breakfast,Primo Thin,
          Pizzaria or Rising Crust Palermo's Pizza WHEN YOU BUY MULTIPLES OF 4 OR 5.99 EA'
          example 3:
          Input text:'11.9 12pk,10 oz Bottles or 12pk,12 oz cans Jack Daniel's Country cocktails'
        ======


        Cluster 3  Below are some examples of cluster 3 images:
        ======
        cluster 3:

        -The text explicitly shows price reduction or discount calculation on the right side.
        -Price is represented as 'X.XX' or follows a pattern like 'X/X$', indicating that the price is unavailable.
        -Image should not have text like $5 Instant savings
          example 1:
          Input:'4.99 When you redeem 4000 pg points 6.99 8k Gatorade'
          example 2:
          Input:'24 ct Food Club Waffles or Pancakes 4.99 SALE PRICE -1.00 When you Buy 5 3.99 ea FINAL COST'
        ======
        """




        self.CLUSTER_1_PROMPT ="""Extract text from the image and categorize it into Size, Pack, RetailAmount, RetailFactor, Description and Ad columns.
              Place values ending in standard weight units (e.g., fl, oz, lb, pc, sf, ltr, inch,ct)
              under 'Size', if no size is given, explicitly assign it to null instead of leaving it empty.

              Assign values with pk to 'Pack' while maintaining the units,
              values associated with keywords (e.g., Mega Roll, Double Roll, gallons) to the 'Pack'. If none of the above conditions match, explicitly assign it to null under the 'Pack' column.

              RetailAmount should contain price-related values like 'Retail Amount' can represent a total price for a specified quantity of items, expressed
              as a fraction where the denominator is the total price and the numerator is the retail factor, when the RetailAmount is associated with ea then
              RetailFactor is one. Single numerical values indicate a RetailAmount equivalent to that value, with a 'Retail Factor' of one. Numerical value with
              a unit of measurement as a suffix indicates a 'Retail Amount' corresponding to that numerical value, where the suffix signifies a 'Size' of
              1 and RetailFactor as 1. If RetailAmount is missing, explicitly assign it to null. If RetailFactor is missing, explicitly assign it to null.

              Keep meaningful product details in 'Description'. Ignore text in bounding boxes for categorization and promotional text or promotional banner is kept under Ad column. If there is no product description should not extract the text.
              If the image contains only promotional or banner text (with no valid product details), place all extracted text into the Ad column, and set all other columns to
             "null". Never leave the output empty
              Ensure accurate extraction and classification. Give the output in json format
.
            Example 1:


            Input Text: '2/$4 18-19 oz Select Old El Paso or Progresso Soup'

            Output:
            {
            "Size": "18-19 oz",
            "Pack": "null",
            "RetailAmount": "$4",
            "RetailFactor": "2",
            "Description": "Old El Paso or Progresso Soup"
            "Ad":null
            }
            Example 2:
            Input Text: '5.99lb 90% Lean Ground Beef'
            Output:
            {
            "Size": "1 lb",
            "Pack": "null",
            "RetailAmount": "$5.99",
            "RetailFactor": "1",
            "Description": "Lean Ground Beef"
            "Ad":null
            }
            # Example 3:
            # Input Text: '2.49lb Boneless Center Cut Pork Chops 3LB OR MORE'
            # Output:
            # {
            # "Size": "1 lb",
            # "Pack": "null",
            # "RetailAmount": "$2.49",
            # "RetailFactor": "3",
            # "Description": "Boneless Center Cut Pork Chops"
            # "Ad":"3LB OR MORE"
            # }

            Example 4:
            Input Text: '$4.00 e Piggly Wiggly's Very Own Rotisserie Chicken with'
            the purchase of two(2)12-16 oz Reser's American Clasic Sides or Salads
            Output:
            {
            "Size": "null",
            "Pack": "null",
            "RetailAmount": "$4.00",
            "RetailFactor": "1",
            "Description": "Piggly Wiggly's Very Own Rotisserie Chicken"
            "Ad":"with the purchase of two(2)12-16 oz Reser's American Clasic Sides or Salads"

"""
 
        self.CLUSTER_2_PROMPT = """Extract text from the image and categorize it into Size, Pack, RetailAmount, RetailFactor, Description and Ad columns.
              Place values ending in standard weight units (e.g., fl, oz, lb, pc, sf, ltr, inch,ct)
              under 'Size', if no size is given, explicitly assign it to null instead of leaving it empty.

              Assign values with pk to 'Pack' while maintaining the units,
              values associated with keywords (e.g., Mega Roll, Double Roll, gallons) to the 'Pack'. If none of the above conditions match, explicitly assign it to null under the 'Pack' column.

              RetailAmount should contain price-related values like 'Retail Amount' can represent a total price for a specified quantity of items, expressed
              as a fraction where the denominator is the total price and the numerator is the retail factor, when the RetailAmount is associated with ea then
              RetailFactor is one. Single numerical values indicate a RetailAmount equivalent to that value, with a 'Retail Factor' of one. Numerical value with
              a unit of measurement as a suffix indicates a 'Retail Amount' corresponding to that numerical value, where the suffix signifies a 'Size' of
              1 and RetailFactor as 1. If RetailAmount is missing, explicitly assign it to null. If RetailFactor is missing, explicitly assign it to null.
              If the text info looks similar to size or unit, but it is associated with keywords (e.g. 'or more', 'above', 'at least', etc...) to represent that is the minimum quantity of products to be purchased, then treat the numeric values as RetailFactor and the unit as the Size.
              Keep meaningful product details in 'Description'. Ignore text in bounding boxes for categorization and promotional text or promotional banner is kept under Ad column. If there is no product description should not extract the text.

              If the image contains only promotional or banner text (with no valid product details), place all extracted text into the Ad column, and set all other columns to
              "null". Never leave the output empty.
              Ensure accurate extraction and classification. Give the output in json format.
              Example 1:

              Input:'4/$5 7-16 oz Barilla Pasta 16-24 oz Ragu Pasta Sauce 1'

              Output:
              [{
              "Size": "7-16 oz",
              "Pack": "null",
              "RetailAmount": "$5",
              "RetailFactor": "4",
              "Description": "Barilla Pasta"
              "Ad":null
              },
              {
              "Size": "16-24 oz",
              "Pack": "null",
              "RetailAmount": "$1.99",
              "RetailFactor": "1",
              "Description": "Ragu Pasta Sauce"
               "Ad":null
              }]
              Example 2:

              Input text:'3LB OR MORE 4.99lb Boneless Sirlloin Tip Roast or Steaks'
              output:
              {
              "Size": "1 lb",
              "Pack": "null",
              "RetailAmount": "$4.99",
              "RetailFactor": "3",
              "Description": "Boneless Sirlloin Tip Roast or Steaks"
              "Ad": "3LB OR MORE"
              }
              Example 3:
              Input :'4.99 10 oz Raybern's Sandwiches or 13-33 oz Breakfast,Primo Thin,
              Pizzaria or Rising Crust Palermo's Pizza WHEN YOU BUY MULTIPLES OF 4 OR 5.99 EA'
              output:
              [{
              "Size": "10 oz"
              "Pack": "null",
              "RetailAmount": "$4.99",
              "RetailFactor": "1",
              "Description": "Raybern's Sandwiches"
              "Ad": "WHEN YOU BUY MULTIPLES OF 4 OR 5.99 EA"
              },
              {
              "Size": "13-33 oz"
              "Pack": "null",
              "RetailAmount": "$4.99",
              "RetailFactor": "1",
              "Description": " Breakfast,Primo Thin Pizzaria or Rising Crust Palermo's Pizza
              "Ad": "WHEN YOU BUY MULTIPLES OF 4 OR 5.99 EA"
              }]
              example 4 :
              Input :'4.99 9.5 oz Gala AngelFood Cake or 6ct Mini Angel Food'
              output:
              {
              "Size": "9.5 oz 6 ct"
              "Pack": "null",
              "RetailAmount": "$4.99",
              "RetailFactor": "1",
              "Description": "Gala AngelFood Cake or Mini Angel Food"
              "Ad": "null"
              }
              example 5
              Input :'11.9 12pk,10 oz Bottles or 12pk,12 oz cans Jack Daniel's Country cocktails'
              output:
              {
              "Size": "10 oz Bottles 12 oz cans"
              "Pack": "12 pk 12 pk",
              "RetailAmount": "$11.99",
              "RetailFactor": "1",
              "Description": "Jack Daniel's Country cocktails"
              "Ad": "null"
              }
"""
        self.CLUSTER_3_PROMPT = """Extract text from the image and categorize it into Size, Pack, RetailAmount, RetailFactor, Description and Ad columns.
              Place values ending in standard weight units (e.g., fl, oz, lb, pc, sf, ltr, inch,ct)
              under 'Size', if no size is given, explicitly assign it to null instead of leaving it empty.

              Assign values with pk to 'Pack' while maintaining the units,
              values associated with keywords (e.g., Mega Roll, Double Roll, gallons) to the 'Pack'. If none of the above conditions match, explicitly assign it to null under the 'Pack' column.

              RetailAmount should contain price-related values like 'Retail Amount' can represent a total price for a specified quantity of items, expressed
              as a fraction where the denominator is the total price and the numerator is the retail factor, when the RetailAmount is associated with ea then
              RetailFactor is one. Single numerical values indicate a RetailAmount equivalent to that value, with a 'Retail Factor' of one. Numerical value with
              a unit of measurement as a suffix indicates a 'Retail Amount' corresponding to that numerical value, where the suffix signifies a 'Size' of
              1 and RetailFactor as 1. If RetailAmount is missing, explicitly assign it to null. If RetailFactor is missing, explicitly assign it to null.
              If the text info looks similar to size or unit, but it is associated with keywords (e.g. 'or more', 'above', 'at least', etc...) to represent that is the minimum quantity of products to be purchased, then treat the numeric values as RetailFactor and the unit as the Size.
              Keep meaningful product details in 'Description'. Ignore text in bounding boxes for categorization and promotional text or promotional banner is kept under Ad column. If there is no product description should not extract the text.
              If the image contains only promotional or banner text (with no valid product details), place all extracted text into the Ad column, and set all other columns to "null". Never leave the output empty.

              Ensure accurate extraction and classification. Give the output in json format

              Example 1:
              Input:4.99 When you redeem 4000 pg points 6.99 8k Gatorade
              output:

              {
              "Size": "7-16 oz",
              "Pack": "8 pk",
              "RetailAmount": "$6.99",
              "RetailFactor": "1",
              "Description": "Gatorade"
              "Ad":"4.99 When you redeem 4000 pg points"
              },
              Example 2:
              Input:24 ct Food Club Waffles or Pancakes 4.99 SALE PRICE -1.00 When you Buy 5 3.99 ea FINAL COST
              output:

              {
              "Size": "24 ct",
              "Pack": "null ",
              "RetailAmount": "$4.99",
              "RetailFactor": "5",
              "Description": "Food Club Waffles or Pancakes"
               "Ad":"4.99 SALE PRICE -1.00 When you Buy 5 "
              }




"""
    # ---------------- Utility ----------------
    # def log_with_time(self, msg: str):
    #     elapsed = datetime.now() - self.START_TIME
    #     m, s = divmod(elapsed.total_seconds(), 60)
    #     print(f"{datetime.now():%Y-%m-%d %H:%M:%S} | ⏱ {int(m)}m {int(s)}s | {msg}")

    def log(self, msg: str):
    
      elapsed = datetime.now() - self.START_TIME
      m, s = divmod(elapsed.total_seconds(), 60)
      print(f"{datetime.now():%Y-%m-%d %H:%M:%S} | ⏱ {int(m)}m {int(s)}s | {msg}")


    def init_gcs_client(self):
        return storage.Client()
    def upload_to_gcs(self, source_file_name, destination_blob_name=None):
        """Uploads a file to GCS."""
        if not destination_blob_name:
            destination_blob_name = self.DESTINATION_BLOB_NAME
        client = storage.Client()
        bucket = client.bucket(self.GCS_BUCKET_NAME)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        self.log(f"✅ File {source_file_name} uploaded to gs://{self.GCS_BUCKET_NAME}/{destination_blob_name}")


    def get_gcs_image_urls(self, extensions=None):
        """Returns a list of GCS URLs for images in the given bucket and prefix."""
        if extensions is None:
            extensions = [".jpg", ".jpeg", ".png"]
        self.log(f"📌 Connecting to bucket: {self.GCS_BUCKET_NAME}")
        client = storage.Client()
        bucket = client.bucket(self.GCS_BUCKET_NAME)
        self.log(f"🔍 Listing blobs with prefix: {self.CLEANED_IMAGES_PREFIX}")
        urls = [
            f"gs://{self.GCS_BUCKET_NAME}/{blob.name}"
            for blob in bucket.list_blobs(prefix=self.CLEANED_IMAGES_PREFIX)
            if any(blob.name.lower().endswith(ext) for ext in extensions)
        ]
        self.log(f"📊 Total images found: {len(urls)}")
        return urls

    # ---------------- Retry Wrapper ----------------
    def retry_function(self, func, max_retries=3, delay=2, backoff=1.5, **kwargs):
        
          image_url = kwargs.get('image_url', 'Unknown')

          for attempt in range(1, max_retries + 1):
              try:
                  result = func(**kwargs)
                  if result:  # success case
                      if attempt > 1:
                          self.log(f"✅ {func.__name__} succeeded for {image_url} on attempt {attempt}")
                      return result
                  else:
                      self.log(f"⚠️ Attempt {attempt} failed: Empty result from {func.__name__} for {image_url}")
              except Exception as e:
                  error_msg = str(e)
                  if any(code in error_msg for code in ["503", "429", "UNAVAILABLE", "Server"]):
                      self.log(f"⚠️ Attempt {attempt} transient server error: {e} for {image_url}")
                  else:
                      self.log(f"⚠️ Attempt {attempt} failed: {e} for {image_url}")

              # Retry with backoff + jitter
              if attempt < max_retries:
                  sleep_time = delay * (backoff ** (attempt - 1))
                  sleep_time += random.uniform(0, 1)  # jitter
                  self.log(f"⏳ Retrying after {sleep_time:.1f}s...")
                  time.sleep(sleep_time)

          # Max retries reached — log failure
          self.log(f"❌ {func.__name__} failed for {image_url} after {max_retries} attempts")

          if func.__name__ == "extract_text":
              self.extraction_failures.append(image_url)
          elif func.__name__ == "classify_image":
              self.classification_failures.append(image_url)

          return None


    # ---------------- Mock classification / extraction (to be replaced with Gemini calls) ----------------
    def classify_image(self, image_url):
        """Classify an image into clusters using Gemini model. Logs only failures."""
        prompt_text = """
        Image:
        Cluster Label:
        """

        response_schema = {
            "type": "object",
            "properties": {
                "clusters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "integer"}
                        },
                        "required": ["label"],
                    },
                }
            }
        }

        classification_prompt = f"{self.CLASSIFICATION_PROMPT}\n{prompt_text}"

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-001",
                contents=[
                    classification_prompt,
                    Part.from_uri(file_uri=image_url, mime_type="image/jpeg")
                ],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": response_schema,
                },
            )
        except Exception as e:
            self.log(f"❌ Classification API call failed for {image_url}: {e}")
            raise

        try:
            response_json = json.loads(response.text)
            label = response_json.get("clusters", [{}])[0].get("label")
            return label
        except Exception as e:
            self.log(f"⚠️ Classification JSON parsing error for {image_url}: {e}")
            return None


    def extract_text(self, image_url, extraction_prompt):
        """Extract structured text from image using Gemini model. Logs only failures."""
        prompt_text = f"Image:\nPrompt: {extraction_prompt}"
        response_schema = {
            "type": "object",
            "properties": {
                "extracted_data": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "Size": {"type": "string"},
                            "Pack": {"type": "string"},
                            "RetailAmount": {"type": "string"},
                            "RetailFactor": {"type": "string"},
                            "Description": {"type": "string"},
                            "Ad": {"type": "string"},
                        },
                        "required": [
                            "Size", "Pack", "RetailAmount",
                            "RetailFactor", "Description", "Ad"
                        ],
                    },
                }
            },
        }

        extraction_prompt_full = f"Extract text from the image\n{prompt_text}"

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-001",
                contents=[
                   extraction_prompt_full,
                   Part.from_uri(file_uri=image_url, mime_type="image/jpeg")
                ],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": response_schema,
                },
            )
        except Exception as e:
            self.log(f"❌ Extraction API call failed for {image_url}: {e}")
            raise

        try:
            response_json = json.loads(response.text)
            return response_json.get("extracted_data", [])
        except Exception as e:
            self.log(f"⚠️ Extraction JSON parsing error for {image_url}: {e}")
            return []


    # ---------------- Processing Logic ----------------
    def process_single_image(self, image_url, bucket_name):
    
      result_rows = []
      filename = os.path.basename(image_url)
      page_number = None

      match = re.search(r'_page_(\d+)', filename)
      if match:
          try:
              page_number = int(match.group(1))
          except ValueError:
              self.log(f"⚠️ Could not parse page number from {filename}")

      # Classification step
      cluster_label = self.retry_function(self.classify_image, image_url=image_url)
      if cluster_label not in [1, 2, 3]:
          cluster_label = self.retry_function(self.classify_image, image_url=image_url)
      if cluster_label not in [1, 2, 3]:
          self.classification_failures.append(image_url)
          return []

      self.classification_success.append((image_url, cluster_label))

      # Select cluster prompt
      extraction_prompt = {
          1: self.CLUSTER_1_PROMPT,
          2: self.CLUSTER_2_PROMPT,
          3: self.CLUSTER_3_PROMPT,
      }.get(cluster_label, self.CLUSTER_1_PROMPT)

      # Extraction step
      extracted_data = self.retry_function(
          self.extract_text, image_url=image_url, extraction_prompt=extraction_prompt
      )

      # Retry on empty/invalid extraction
      def is_row_empty(row):
          return all(
              (val is None or str(val).strip() == "" or str(val).upper() == "NULL")
              for val in row.values()
          )

      if not extracted_data or any(is_row_empty(row) for row in extracted_data):
          extracted_data = self.retry_function(
              self.extract_text, image_url=image_url, extraction_prompt=extraction_prompt
          )

      if not extracted_data or any(is_row_empty(row) for row in extracted_data):
          self.extraction_failures.append(image_url)
          self.log(f"🚨 Skipping {image_url} — extraction failed")
          return []

      # Clean & append metadata
      for row in extracted_data:
          for key, value in row.items():
              if isinstance(value, str):
                  row[key] = re.sub(r"\s+", " ", value.strip())
          row["Page_Number"] = page_number
          row["Cluster"] = cluster_label
          result_rows.append(row)

      self.extraction_success.append(image_url)
      return result_rows

    def process_images(self,gcs_urls, thread_count=None):
      if thread_count is None:
        thread_count = self.THREAD_COUNT
    

      all_results = []

      self.log(f"🚀 Starting processing {len(self.gcs_urls)} images with {thread_count} threads...")

      with ThreadPoolExecutor(max_workers=thread_count) as executor:
          futures = [executor.submit(self.process_single_image, url, self.GCS_BUCKET_NAME) for url in self.gcs_urls]

          for future in as_completed(futures):
              try:
                  result_rows = future.result()
                  if result_rows:
                      all_results.extend(result_rows)
              except Exception as e:
                  self.log(f"❌ Exception in ThreadPool: {e}")

      # Convert all extracted results to DataFrame
      self.df6 = pd.DataFrame(all_results, columns=[
          "Size", "Pack", "RetailAmount", "RetailFactor",
          "Description", "Ad", "Page_Number", "Cluster"
      ])

      self.log(f"✅ Completed processing {len(self.gcs_urls)} images.")
      self.log(f"📊 Final DataFrame contains {len(self.df6)} rows.")

      return self.df6





  


    

    # ---------------- Main Pipeline ----------------
    def run(self):
      pipeline_start = time.time()

      self.log("🔹 Starting Text Extraction Pipeline...")

      gcs_urls = self.get_gcs_image_urls()
      if not gcs_urls:
          self.log("⚠️ No images found for extraction.")
          return
      self.gcs_urls = gcs_urls

      # Step 1: Run Text Extraction
      final_results = self.process_images(self.gcs_urls[0:5])

      # Step 2: Filter out empty rows
      cols_to_check = ["Size", "Pack", "RetailAmount", "RetailFactor", "Description"]
      mask = final_results[cols_to_check].apply(
          lambda row: all(
              (val is None or str(val).strip() == "" or str(val).upper() == "NULL")
              for val in row
          ),
          axis=1
      )
      final_results = final_results[~mask].reset_index(drop=True)

      # Step 3: Save and upload results
      final_results.to_excel(self.OUTPUT_FILE, index=False, na_rep="")
      self.upload_to_gcs(self.OUTPUT_FILE, self.DESTINATION_BLOB_NAME)

      total_time = time.time() - pipeline_start
      self.log(f"🏁 Text Extraction complete | Total time: {total_time:.2f}s")
