import os
import re
import time
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from google.cloud import storage
import google.genai as genai
from io import BytesIO

os.environ["TF_CPP_MINlog_LEVEL"] = "3"  # Suppress TensorFlow INFO and WARNING messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN custom ops for consistent results

import tensorflow as tf


class LLMSimilarityComponent:
    def __init__(self, bucket: str, input_pdf: str, master_excel_filename: str,verbose: bool = True):
        # -------------------------
        # CONFIGURATION
        # -------------------------
        self.GCS_BUCKET_NAME = bucket
        self.INPUT_PDF = input_pdf
        self.MASTER_FILENAME = master_excel_filename

        # Root path structure
        self.GCS_PIPELINE_ROOT_PREFIX = "DPAC/6.pipelineRun"
        pdf_name_only = os.path.splitext(os.path.basename(self.INPUT_PDF))[0]
        self.PDF_ROOT_PREFIX = f"{self.GCS_PIPELINE_ROOT_PREFIX}/{pdf_name_only}"

        # File paths
        self.INPUT_FILE_PATH = f"{self.PDF_ROOT_PREFIX}/NLP_Results.xlsx"
        self.OUTPUT_FILE_LLM = f"{self.PDF_ROOT_PREFIX}/LLM_Results.xlsx"

        # Vertex AI / GenAI
        self.VERTEXAI_PROJECT = "1039491753481"
        self.VERTEXAI_LOCATION = "us-east4"
        self.ENDPOINT = (
            "projects/1039491753481/locations/us-east4/endpoints/4302428581906087936"
        )

        # Parallelization and retry settings
        self.MAX_WORKERS = 100         # Keep high concurrency
        self.RETRY_WORKERS = 20
        self.PRINT_INTERVAL = 5000
        self.RETRY_ROUNDS = 1          # Retry attempts
        self.RETRY_BACKOFF_SECONDS = 2.0
        self.PER_TASK_TIMEOUT = 45     # Timeout per LLM call (seconds)
        self.AS_COMPLETED_TIMEOUT = 180  # Timeout for as_completed()
        self.THROTTLE_INTERVAL = 0.02  # Small delay every few submissions
        self.verbose=verbose

        # Client setup
        self.client = genai.Client(
            vertexai=True,
            project=self.VERTEXAI_PROJECT,
            location=self.VERTEXAI_LOCATION,
        )

        self.PIPELINE_START = datetime.now()

    # --------------------------
    # LOGGING
    # --------------------------
    def _timestamp(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def log(self, msg: str):
      if self.verbose:  # only log when verbose=True
        elapsed = datetime.now() - self.PIPELINE_START
        m, s = divmod(elapsed.total_seconds(), 60)
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} | ⏱ {int(m)}m {int(s)}s | {msg}")
    
    # --------------------------
    # GCS HELPERS
    # --------------------------
    def _load_excel_from_gcs(self, file_path: str):
        client_gcs = storage.Client()
        bucket = client_gcs.bucket(self.GCS_BUCKET_NAME)
        blob = bucket.blob(file_path)
        content = blob.download_as_bytes()
        df = pd.read_excel(BytesIO(content))
        df.columns = df.columns.str.strip()
        self.log(f"📥 Loaded '{file_path}' from GCS. Shape: {df.shape}")
        return df

    def _upload_to_gcs(self, file_path: str, df: pd.DataFrame):
        client_gcs = storage.Client()
        bucket = client_gcs.bucket(self.GCS_BUCKET_NAME)
        blob = bucket.blob(file_path)
        buffer = BytesIO()
        df.to_excel(buffer, index=False)
        buffer.seek(0)
        blob.upload_from_file(
            buffer,
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        self.log(f"✅ Uploaded results to gs://{self.GCS_BUCKET_NAME}/{file_path}")

    # --------------------------
    # CORE LLM CALL
    # --------------------------
    def _get_similarity_score(self, index, row):
        prompt = f"""Compare the following two descriptions and return a similarity score between 0 and 1:
        Description A: {row['PDF_Description']}
        Description B: {row['Excel_Description']}"""

        start_time = time.time()
        try:
            response = self.client.models.generate_content(
                model=self.ENDPOINT,
                contents=prompt,
            )
            match = re.search(r"[-+]?[0-9]*\.?[0-9]+", response.text)
            score = float(match.group()) if match else None
            latency = time.time() - start_time
            return index, score, latency
        except Exception:
            return index, None, None

    def _wrapped_get_similarity(self, idx, row, total_len):
        index, score, latency = self._get_similarity_score(idx, row)
        if (idx + 1) % self.PRINT_INTERVAL == 0 or (idx + 1) == total_len:
            self.log(f"🟢 Completed {idx + 1}/{total_len} rows")
        return index, score, latency

    # --------------------------
    # PARALLEL EXECUTION (with timeout + micro-throttle)
    # --------------------------
    def _parallel_similarity_scoring(self, df: pd.DataFrame, max_workers: int):
        self.log(f"\n🚀 Starting similarity scoring for {len(df)} rows using {max_workers} threads...")

        df = df.copy()
        results = {}
        latencies = []
        failed_rows = []

        start_time = datetime.now()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for idx, row in df.iterrows():
                futures[executor.submit(self._wrapped_get_similarity, idx, row, len(df))] = idx
                if idx % 5 == 0:
                    time.sleep(self.THROTTLE_INTERVAL)

            for future in as_completed(futures, timeout=self.AS_COMPLETED_TIMEOUT):
                idx = futures[future]
                try:
                    result = future.result(timeout=self.PER_TASK_TIMEOUT)
                    index, score, latency = result
                    results[index] = score
                    if latency:
                        latencies.append(latency)
                except TimeoutError:
                    self.log(f"⚠️ Timeout for index {idx}")
                    failed_rows.append(idx)
                except Exception as e:
                    self.log(f"⚠️ Exception for index {idx}: {e}")
                    failed_rows.append(idx)

        total_time = (datetime.now() - start_time).total_seconds()
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        self.log(f"🧠 Completed batch in {total_time:.2f}s | Avg latency: {avg_latency:.2f}s | Failed: {len(failed_rows)}")

        df["LLM_Similarity"] = df.index.map(results)
        return df

    # --------------------------
    # MAIN RUN PIPELINE
    # --------------------------
    def run(self):
        start_time = datetime.now()
        self.log("📥 Loading input data from GCS...")

        df = self._load_excel_from_gcs(self.INPUT_FILE_PATH)

        self.log("🔍 Generating initial similarity scores...")
        df = self._parallel_similarity_scoring(df, max_workers=self.MAX_WORKERS)

        # ======================
        # Retry logic
        # ======================
        failed_mask = df["LLM_Similarity"].isnull() | df["LLM_Similarity"].eq("")
        failed_df = df[failed_mask]

        if not failed_df.empty:
            self.log(f"🔁 Retrying {len(failed_df)} failed rows...")
            retry_start = datetime.now()

            pending = failed_df.copy()

            for attempt in range(1, self.RETRY_ROUNDS + 1):
                self.log(f"  ↺ Retry attempt {attempt} with {len(pending)} rows...")
                retried_df = self._parallel_similarity_scoring(
                    pending, max_workers=self.RETRY_WORKERS
                )

                filled = 0
                for idx, score in retried_df["LLM_Similarity"].items():
                    if pd.notna(score):
                        df.at[idx, "LLM_Similarity"] = score
                        filled += 1

                self.log(f"    ✅ Filled {filled} scores on attempt {attempt}")

                pending_idx = [i for i in pending.index if pd.isna(df.at[i, "LLM_Similarity"])]
                if not pending_idx:
                    self.log("🎯 All retries succeeded.")
                    break

                pending = pending.loc[pending_idx]

                if attempt < self.RETRY_ROUNDS:
                    delay = self.RETRY_BACKOFF_SECONDS * attempt
                    self.log(f"⏳ Waiting {delay:.1f}s before next attempt...")
                    time.sleep(delay)

            retry_time = (datetime.now() - retry_start).total_seconds()
            self.log(f"🔄 All retry rounds finished in {retry_time:.2f}s")
        else:
            self.log("✅ No failed rows to retry.")

        # Upload results
        self._upload_to_gcs(self.OUTPUT_FILE_LLM, df)

        total_time = (datetime.now() - start_time).total_seconds()
        minutes, seconds = divmod(total_time, 60)
        self.log(f"🏁 LLM Similarity step completed in {int(minutes)}m {seconds:.2f}s")

        return df
