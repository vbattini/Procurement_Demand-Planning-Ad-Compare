# full_pipeline_with_brand_fuzzy_latest.py
import os
import re
import io
import sys
import time
import json
import random
import hashlib
from datetime import datetime
from typing import List, Tuple, Any

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
from google.cloud import storage

# Optional GenAI imports (kept but not required for core pipeline)
try:
    from google import genai  # optional; not used by core pipeline
except Exception:
    genai = None


class NLPMatchingComponent:
    def __init__(self, bucket: str, input_pdf: str, master_excel_filename: str,verbose: bool = True):
        # -------------------------
        # CONFIGURATION
        # -------------------------
        self.bucket_name = bucket
        self.INPUT_PDF = input_pdf
        self.MASTER_FILENAME = master_excel_filename

        # -------------------------
        # PATHS
        # -------------------------
        self.GCS_PIPELINE_ROOT_PREFIX = "DPAC/6.pipelineRun"
        self.pdf_name_only = os.path.splitext(os.path.basename(self.INPUT_PDF))[0]
        self.PDF_ROOT_PREFIX = f"{self.GCS_PIPELINE_ROOT_PREFIX}/{self.pdf_name_only}"

        self.COMPARE_FILE = f"{self.PDF_ROOT_PREFIX}/Text_Extraction.xlsx"
        self.MASTER_FILE = f"{self.GCS_PIPELINE_ROOT_PREFIX}/{self.MASTER_FILENAME}"
        self.OUTPUT_FILE_NLP = f"{self.PDF_ROOT_PREFIX}/NLP_Results.xlsx"
        self.verbose=verbose

        self.PIPELINE_START = datetime.now()

        # -------------------------
        # COLUMNS
        # -------------------------
        self.PDF_COLUMNS = ["Description", "Size", "Pack", "Retail Factor", "Retail Amt", "Page_Number"]
        self.Excel_COLUMNS = ["Description", "Size", "Pack", "Retail Factor", "Retail Amt", "nCat Desc", "BRAND"]

        # -------------------------
        # NLP PARAMETERS
        # -------------------------
        self.BOOST_THRESHOLD = 0.2
        self.TOP_K = 65
        self.MODEL_NAME = "all-MiniLM-L6-v2"

        # Regex patterns
        self.VOWELS = set("aeiou")
        self.RE_NUMERIC = re.compile(r"[^0-9.]")
        self.RE_WORDS = re.compile(r"[A-Za-z]+")
        self.RE_UPPERCASE = re.compile(r"[A-Z]")

        # GCS client placeholder
        self._gcs_client = None

    # -------------------------
    # LOGGING
    # -------------------------
    def log(self, msg: str):
      if self.verbose:  # only log when verbose=True
        elapsed = datetime.now() - self.PIPELINE_START
        m, s = divmod(elapsed.total_seconds(), 60)
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} | ⏱ {int(m)}m {int(s)}s | {msg}")
    # -------------------------
    # GCS HELPERS
    # -------------------------
    def get_gcs_client(self):
        if self._gcs_client is None:
            self._gcs_client = storage.Client()
        return self._gcs_client

    def load_excel_from_gcs(self, file_path: str, usecols: List[str] = None) -> pd.DataFrame:
        client = self.get_gcs_client()
        bucket = client.bucket(self.bucket_name)
        blob = bucket.blob(file_path)
        if not blob.exists():
            raise FileNotFoundError(f"Blob {file_path} not found in bucket {self.bucket_name}")
        content = blob.download_as_bytes()
        df = pd.read_excel(io.BytesIO(content), usecols=usecols)
        df.columns = df.columns.str.strip()
        self.log(f"Loaded '{file_path}' from GCS. Shape: {df.shape}")
        return df

    def upload_to_gcs(self, file_path: str, data: pd.DataFrame):
        client = self.get_gcs_client()
        bucket = client.bucket(self.bucket_name)
        blob = bucket.blob(file_path)
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            data.to_excel(writer, index=False)
        buffer.seek(0)
        blob.upload_from_file(buffer, content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        self.log(f"✅ Uploaded results to gs://{self.bucket_name}/{file_path}")

    # -------------------------
    # CLEANING
    # -------------------------
    def clean_dataframe(self, df: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
        self.log(f"Before cleaning: {df.shape}")
        df = df.copy()
        df.columns = df.columns.str.strip()
        str_cols = df.select_dtypes(include=["object"]).columns
        if len(str_cols) > 0:
            df[str_cols] = df[str_cols].astype(str).apply(lambda s: s.str.strip())
        present_required = [c for c in required_columns if c in df.columns]
        if present_required:
            df = df.drop_duplicates(subset=present_required).reset_index(drop=True)
        df = df.replace(["", "nan", "NaT", "null", "NULL", "None", None], pd.NA).fillna("Null")

        def all_required_null(row):
            for x in row:
                if not (str(x).strip().upper() in ("NULL", "NAN", "NONE", "")):
                    return False
            return True

        if present_required:
            df = df[~df[present_required].apply(all_required_null, axis=1)]
        self.log(f"After cleaning: {df.shape}")
        return df.reset_index(drop=True)

    def deduplicate_description(self, df2: pd.DataFrame) -> pd.DataFrame:
        self.log("Deduplicating master descriptions...")
        columns_to_keep = ["Description", "Size", "Pack", "Retail Factor", "Retail Amt", "nCat Desc", "BRAND"]
        cols_present = [c for c in columns_to_keep if c in df2.columns]
        df2_trimmed = df2[cols_present].copy()
        dedup_subset = ["Description"]
        if "nCat Desc" in df2_trimmed.columns:
            dedup_subset.append("nCat Desc")
        if "BRAND" in df2_trimmed.columns:
            dedup_subset.append("BRAND")
        deduped = df2_trimmed.drop_duplicates(subset=dedup_subset).reset_index(drop=True)
        self.log(f"Deduplicated master dataset size: {len(deduped)}")
        return deduped

    # -------------------------
    # TEXT PROCESSING HELPERS
    # -------------------------
    def clean_text_start(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""
        text = text.strip()
        i = 0
        while i < len(text) and not text[i].isalnum():
            i += 1
        return text[i:]

    def devoweled_abbreviation_with_prefix(self, text: str, excel_desc: str = "") -> str:
        text = self.clean_text_start(text)
        if not text:
            return ""
        words = self.RE_WORDS.findall(text)
        if not words:
            return ""
        processed = "".join(w[0] + "".join(ch for ch in w[1:] if ch.lower() not in self.VOWELS) for w in words)
        if isinstance(excel_desc, str) and excel_desc.strip():
            excel_words = self.RE_WORDS.findall(excel_desc)
            if excel_words:
                excel_first = excel_words[0].lower()
                for i in range(3, len(excel_first) + 1):
                    if processed.lower().startswith(excel_first[:i]):
                        return excel_first[:i] + processed[i:]
        return processed

    def uppercase_abbreviation_with_prefix(self, text: str, excel_desc: str = "") -> str:
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        if not text.strip():
            return ""
        abbr = "".join(self.RE_UPPERCASE.findall(text))
        prefix_len = 0
        if isinstance(excel_desc, str) and excel_desc.strip():
            first_word = excel_desc.strip().split()[0]
            prefix_len = len(first_word)
        if prefix_len > 0:
            abbr = abbr.ljust(prefix_len, "_")[:prefix_len]
        return abbr

    def uppercase_abbreviation_match(self, desc1: str, desc2: str, excel_desc: str = "") -> bool:
        abbr1 = self.uppercase_abbreviation_with_prefix(desc1, excel_desc)
        abbr2 = self.uppercase_abbreviation_with_prefix(desc2, excel_desc)
        return abbr1 == abbr2

    def first_char_match(self, desc1: str, desc2: str) -> bool:
        desc1 = self.clean_text_start(desc1)
        desc2 = self.clean_text_start(desc2)
        return bool(desc1 and desc2 and desc1[0].lower() == desc2[0].lower())

    def word_by_word_cross_column_match(self, c_text: str, m_text: str):
        c_words = self.RE_WORDS.findall(self.clean_text_start(c_text))
        m_words = self.RE_WORDS.findall(self.clean_text_start(m_text))
        if not c_words or not m_words:
            return {"match_ratio": 0, "matches": []}
        c_set = {w.lower() for w in c_words}
        c_set |= {self.devoweled_abbreviation_with_prefix(w).lower() for w in c_words}
        # additional devowel variants
        def strict_devowel(w: str) -> str:
            return w[0] + "".join(ch for ch in w[1:] if ch.lower() not in self.VOWELS) if len(w) > 1 else w

        def devowel_keep_last(w: str) -> str:
            if len(w) <= 2:
                return w
            first, middle, last = w[0], w[1:-1], w[-1]
            middle_no_vowels = "".join(ch for ch in middle if ch.lower() not in self.VOWELS)
            return first + middle_no_vowels + last

        for w in c_words:
            c_set.add(strict_devowel(w).lower())
            c_set.add(devowel_keep_last(w).lower())

        m_set = {w.lower() for w in m_words}
        intersection = c_set & m_set
        return {"match_ratio": len(intersection), "matches": list(intersection)}

    def abbreviation_boost_vectorized(self, desc1_array: np.ndarray, desc2_array: np.ndarray,
                                      devowel_weight=0.4, upper_weight=0.4,
                                      first_char_weight=0.2, word_ratio_weight=0.5) -> Tuple[np.ndarray, ...]:
        boosts, dev_scores, upper_scores, first_scores, word_scores = [], [], [], [], []
        for d1, d2 in zip(desc1_array, desc2_array):
            dv1 = self.devoweled_abbreviation_with_prefix(d1, d2).lower()
            dv2 = self.clean_text_start(d2).lower()
            dev_sim = int(dv1[:3] == dv2[:3]) if dv1 and dv2 else 0
            upper_sim = int(self.uppercase_abbreviation_match(d1, d2, d2))
            first_sim = int(self.first_char_match(d1, d2))
            word_ratio = self.word_by_word_cross_column_match(d1, d2)["match_ratio"]
            boost = (devowel_weight * dev_sim +
                     upper_weight * upper_sim +
                     first_char_weight * first_sim +
                     word_ratio_weight * word_ratio)
            boosts.append(boost)
            dev_scores.append(dev_sim)
            upper_scores.append(upper_sim)
            first_scores.append(first_sim)
            word_scores.append(word_ratio)
        return (np.array(boosts, dtype=np.float32),
                np.array(dev_scores, dtype=np.float32),
                np.array(upper_scores, dtype=np.float32),
                np.array(first_scores, dtype=np.float32),
                np.array(word_scores, dtype=np.float32))

    # -------------------------
    # FUZZY MATCH SCORE HELPER
    # -------------------------
    def text_to_word_list(self, text: str) -> List[str]:
        if not text or not isinstance(text, str):
            return []
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        return re.findall(r'\w+', text)

    def match_score(self, PDF_desc: str, Excel_field: Any, mode: str = "WRatio", threshold: float = 0.0) -> Tuple[float, List[Tuple[str, str, str, float]]]:
        if not PDF_desc or not Excel_field:
            return 0.0, []

        def clean_text(t: str) -> str:
            t = str(t).lower().strip()
            t = re.sub(r'[^a-z0-9\s]', '', t)
            return t

        PDF_clean = clean_text(PDF_desc)
        if isinstance(Excel_field, str):
            Excel_clean = clean_text(Excel_field)
        elif isinstance(Excel_field, list):
            Excel_clean = clean_text(" ".join([str(x) for x in Excel_field if isinstance(x, str)]))
        else:
            Excel_clean = clean_text(str(Excel_field))

        if not Excel_clean or not PDF_clean:
            return 0.0, []

        match_info = []

        # exact substring
        if Excel_clean in PDF_clean:
            match_info.append(("full_string", Excel_clean, PDF_clean, 1.0))
            return 1.0, match_info

        if mode == "partial_ratio":
            score = fuzz.partial_ratio(Excel_clean, PDF_clean) / 100.0
        else:
            score = fuzz.WRatio(Excel_clean, PDF_clean) / 100.0

        match_info.append((mode, Excel_clean, PDF_clean, score))
        return (score if score >= threshold else 0.0), match_info

    # -------------------------
    # CANDIDATE GENERATION
    # -------------------------
    def create_combinations(self, df1: pd.DataFrame, df2_desc_deduped: pd.DataFrame, boost_threshold: float = None) -> pd.DataFrame:
        self.log("Creating candidate combinations (full cartesian)...")
        if df1 is None or df1.empty or df2_desc_deduped is None or df2_desc_deduped.empty:
            self.log("One of the inputs is empty; returning empty DataFrame")
            return pd.DataFrame()

        df1_prefixed = df1.add_prefix("PDF_")
        df2_prefixed = df2_desc_deduped.add_prefix("Excel_")
        combined = df1_prefixed.assign(_key=1).merge(df2_prefixed.assign(_key=1), on="_key").drop("_key", axis=1)
        self.log(f"Candidate pairs generated: {len(combined)}")

        # compute desc boost (and BRAND boost)
        desc_boost, *_ = self.abbreviation_boost_vectorized(
            combined["PDF_Description"].to_numpy(),
            combined["Excel_Description"].to_numpy()
        )

        if "Excel_BRAND" in combined.columns:
            brand_boost, *_ = self.abbreviation_boost_vectorized(
                combined["PDF_Description"].to_numpy(),
                combined["Excel_BRAND"].to_numpy()
            )
        else:
            brand_boost = np.zeros(len(combined), dtype=np.float32)

        combined["Boost_Score"] = desc_boost + brand_boost

        if boost_threshold is not None:
            before = len(combined)
            combined = combined.loc[combined["Boost_Score"] >= boost_threshold].reset_index(drop=True)
            self.log(f"Filtered candidates: {len(combined)} (threshold={boost_threshold}) from {before}")

        return combined

    # -------------------------
    # SBERT SIMILARITY
    # -------------------------
    def compute_sbert_similarity(self, df: pd.DataFrame, model_name: str = None, batch_size: int = 512, device: str = None, top_k: int = None) -> pd.DataFrame:
        if df is None or df.empty:
            self.log("No candidates to compute similarity.")
            return pd.DataFrame()

        model_name = model_name or self.MODEL_NAME
        top_k = top_k if top_k is not None else self.TOP_K
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.log(f"Loading SBERT model '{model_name}' on device '{device}'...")
        model = SentenceTransformer(model_name, device=device)

        # Ensure strings
        df = df.fillna("").astype(str)

        # Build unique lists for embeddings
        PDF_texts = df["PDF_Description"].dropna().unique().tolist()
        Excel_descs = df["Excel_Description"].dropna().unique().tolist() if "Excel_Description" in df.columns else []
        Excel_BRANDs = df["Excel_BRAND"].dropna().unique().tolist() if "Excel_BRAND" in df.columns else []
        Excel_ncates = df["Excel_nCat Desc"].dropna().unique().tolist() if "Excel_nCat Desc" in df.columns else []

        # Encode and normalize
        PDF_emb = torch.nn.functional.normalize(
            model.encode(PDF_texts, convert_to_tensor=True, batch_size=batch_size, show_progress_bar=False),
            p=2, dim=1
        )
        Excel_desc_emb = torch.nn.functional.normalize(
            model.encode(Excel_descs, convert_to_tensor=True, batch_size=batch_size, show_progress_bar=False),
            p=2, dim=1
        )
        Excel_BRAND_emb = torch.nn.functional.normalize(
            model.encode(Excel_BRANDs, convert_to_tensor=True, batch_size=batch_size, show_progress_bar=False),
            p=2, dim=1
        ) if Excel_BRANDs else None
        Excel_ncat_emb = torch.nn.functional.normalize(
            model.encode(Excel_ncates, convert_to_tensor=True, batch_size=batch_size, show_progress_bar=False),
            p=2, dim=1
        ) if Excel_ncates else None

        PDF_map = dict(zip(PDF_texts, PDF_emb))
        Excel_map = dict(zip(Excel_descs, Excel_desc_emb))
        Excel_BRAND_map = dict(zip(Excel_BRANDs, Excel_BRAND_emb)) if Excel_BRAND_emb is not None else {}
        Excel_ncat_map = dict(zip(Excel_ncates, Excel_ncat_emb)) if Excel_ncat_emb is not None else {}

        def safe_dot(a, b):
            try:
                if a is None or b is None:
                    return 0.0
                return float(torch.dot(a, b).item())
            except Exception:
                return 0.0

        Description_Similarity = np.array([safe_dot(PDF_map.get(c), Excel_map.get(m)) for c, m in zip(df["PDF_Description"], df["Excel_Description"])])
        Brand_Description_Similarity = np.array([safe_dot(PDF_map.get(c), Excel_BRAND_map.get(b)) for c, b in zip(df["PDF_Description"], df.get("Excel_BRAND", [""] * len(df)) )])
        Ncat_Description_Similarity = np.array([safe_dot(PDF_map.get(c), Excel_ncat_map.get(n)) for c, n in zip(df["PDF_Description"], df.get("Excel_nCat Desc", [""] * len(df)) )])

        df["Description_Similarity"] = pd.Series(Description_Similarity).fillna(0.0)
        df["Brand_Description_Similarity"] = pd.Series(Brand_Description_Similarity).fillna(0.0)
        df["Ncat_Description_Similarity"] = pd.Series(Ncat_Description_Similarity).fillna(0.0)
        df["Boost_Score"] = pd.to_numeric(df.get("Boost_Score", 0), errors="coerce").fillna(0)

        # word lists
        df["Excel_brand_words"] = df["Excel_BRAND"].apply(self.text_to_word_list) if "Excel_BRAND" in df.columns else [[] for _ in range(len(df))]
        df["Excel_ncat_words"] = df["Excel_nCat Desc"].apply(self.text_to_word_list) if "Excel_nCat Desc" in df.columns else [[] for _ in range(len(df))]

        # fuzzy/substring for brand and ncat
        df["Brand_Word_Score"] = df.apply(lambda row: self.match_score(row["PDF_Description"], row.get("Excel_BRAND", ""), mode="partial_ratio")[0], axis=1)
        df["ncat_word_score"] = df.apply(lambda row: self.match_score(row["PDF_Description"], row.get("Excel_nCat Desc", ""), mode="WRatio")[0], axis=1)

        df["total_Combined_Score"] = (
            df["Description_Similarity"].astype(float) +
            df["Ncat_Description_Similarity"].astype(float) +
            df["Brand_Description_Similarity"].astype(float) +
            df["Boost_Score"].astype(float) +
            df["Brand_Word_Score"].astype(float) +
            df["ncat_word_score"].astype(float)
        )

        # cleanup temporary columns
        df.drop(columns=["Excel_brand_words", "Excel_ncat_words"], inplace=True, errors="ignore")

        # scale Combined_Score between 0 and 1
        min_score = df["total_Combined_Score"].min()
        max_score = df["total_Combined_Score"].max()
        if pd.isna(min_score) or pd.isna(max_score):
            df["Combined_Score"] = 0.0
        elif max_score > min_score:
            df["Combined_Score"] = (df["total_Combined_Score"] - min_score) / (max_score - min_score)
        else:
            df["Combined_Score"] = 0.0

        # Top-K per PDF_Description
        if not np.isnan(top_k):
            try:
                df_topk = (
                    df.sort_values(["PDF_Description", "total_Combined_Score"], ascending=[True, False])
                    .groupby("PDF_Description", group_keys=False)
                    .head(int(top_k))
                    .reset_index(drop=True)
                )
            except Exception:
                df_topk = df.reset_index(drop=True)
        else:
            df_topk = df.reset_index(drop=True)

        self.log(f"Top-K candidate rows returned: {len(df_topk)}")
        return df_topk

    # -------------------------
    # PIPELINE RUN
    # -------------------------
    def run(self) -> pd.DataFrame:
        self.log("🚀 NLP Matching pipeline started")

        # You can override compare_path as needed
        compare_path = self.COMPARE_FILE

        try:
            df1 = self.load_excel_from_gcs(compare_path)
            df1 = df1.rename(columns={
                "RetailAmount": "Retail Amt",
                "RetailFactor": "Retail Factor",
                "Page_Number": "Page_Number"
            })

            df2 = self.load_excel_from_gcs(self.MASTER_FILE, usecols=self.Excel_COLUMNS)
        except Exception as e:
            self.log(f"❌ Error loading files: {e}")
            return pd.DataFrame()

        # normalize/rename if needed
        # ensure Description column exists and filter out Nulls
        if "Description" not in df1.columns:
            self.log("❌ Compare file missing 'Description' column.")
            return pd.DataFrame()

        df1 = df1[df1["Description"].astype(str).str.strip().str.lower() != "null"].reset_index(drop=True)
        df1_clean = self.clean_dataframe(df1, self.PDF_COLUMNS)
        df1_clean = df1_clean[df1_clean["Description"].notna()]
        df1_clean = df1_clean[df1_clean["Description"].astype(str).str.strip() != ""]
        df1_clean = df1_clean[df1_clean["Description"].astype(str).str.strip().str.lower() != "null"]
        self.log(f"Filtered compare rows with valid Description: {df1_clean.shape}")

        df2_clean = self.clean_dataframe(df2, self.Excel_COLUMNS)
        df2_deduped = self.deduplicate_description(df2_clean)

        combined = self.create_combinations(df1_clean, df2_deduped, boost_threshold=self.BOOST_THRESHOLD)
        if combined.empty:
            self.log("No candidate pairs after combination/boost filtering.")
            final_results = pd.DataFrame()
        else:
            final_results = self.compute_sbert_similarity(combined, model_name=self.MODEL_NAME, top_k=self.TOP_K)

        # upload results
        try:
            self.upload_to_gcs(self.OUTPUT_FILE_NLP, final_results)
        except Exception as e:
            self.log(f"⚠️ Upload failed: {e}")

        self.log("✅ NLP Comparison completed successfully.")
        return final_results


# -------------------------
# MAIN wrapper
# -------------------------
def main():
    # Read config from environment variables or use example defaults
    bucket = os.environ.get("GCS_BUCKET_NAME", "gc-us-gcs-aiml-dev")
    input_pdf = os.environ.get("INPUT_PDF", "PWMW_B&W_9.9.25_Ad_Test.pdf")
    master_excel = os.environ.get("MASTER_EXCEL", "PWMW_Master.xlsx")
    # Construct component and run
    comp = NLPMatchingComponent(bucket=bucket, input_pdf=input_pdf, master_excel_filename=master_excel)
    results = comp.run()
    # Print a short summary
    if results is None or results.empty:
        print("No results generated.")
    else:
        print(f"Results shape: {results.shape}")
        # show top 5 rows
        print(results.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
