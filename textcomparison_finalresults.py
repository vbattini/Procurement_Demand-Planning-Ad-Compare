# ===========================
# final_match_component.py
# ===========================
import os
import re
import io
import sys
import time
import json
import base64
import random
import hashlib
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import storage
# from sentence_transformers import SentenceTransformer  # keep if you use SBERT elsewhere
from rapidfuzz import fuzz
import google.genai as genai

# ===========================
# GENAI CONFIG (optional)
# ===========================
PROJECT_ID = "gc-proj-aiml-dev-01fd"
if not PROJECT_ID or PROJECT_ID == "[your-project-id]":
    PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT", ""))

LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
try:
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
except Exception:
    client = None  # optional; keep safe if genai not configured

# ===========================
# FINAL MATCH COMPONENT
# ===========================
class FinalMatchComponent:
    def __init__(self, bucket_name, pdf_input_path, master_file_path, pipeline_root_prefix="DPAC/6.pipelineRun",verbose: bool = True):
        self.GCS_BUCKET_NAME = bucket_name
        self.pdf_input_path = pdf_input_path
        self.master_file_path = master_file_path

        self.GCS_PIPELINE_ROOT_PREFIX = pipeline_root_prefix
        pdf_name_only = os.path.splitext(os.path.basename(self.pdf_input_path))[0]
        self.PDF_ROOT_PREFIX = f"{self.GCS_PIPELINE_ROOT_PREFIX}/{pdf_name_only}"

        self.INPUT_FILE_PATH = f"{self.PDF_ROOT_PREFIX}/LLM_Results.xlsx"
        self.OUTPUT_FILE_PATH = f"{self.PDF_ROOT_PREFIX}/Comparison_Results.xlsx"
        self.EXCEL_FILE_PATH = f"{self.GCS_PIPELINE_ROOT_PREFIX}/{self.master_file_path}"
        self.verbose=verbose

        self.PIPELINE_START = datetime.now()

    # -------------------------
    # LOGGING
    # -------------------------
    def log(self, msg: str):
      if self.verbose:  # only log when verbose=True
        elapsed = datetime.now() - self.PIPELINE_START
        m, s = divmod(elapsed.total_seconds(), 60)
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} | ⏱ {int(m)}m {int(s)}s | {msg}")
    
    def normalize_columns(self, df):
        """Replace spaces with underscores and strip column names for consistent access."""
        df = df.copy()
        df.columns = (
            df.columns.astype(str)
            .str.strip()
            .str.replace(" ", "_")
            .str.replace(r"[^\w_]", "", regex=True)
        )
        return df
    # -------------------------
    # GCS HELPERS
    # -------------------------
    def load_excel_from_gcs(self, file_path, usecols=None):
        client = storage.Client()
        bucket = client.bucket(self.GCS_BUCKET_NAME)
        blob = bucket.blob(file_path)
        content = blob.download_as_bytes()
        df = pd.read_excel(io.BytesIO(content), usecols=usecols)
        df.columns = df.columns.str.strip()
        self.log(f"Loaded '{file_path}' from GCS. Shape: {df.shape}")
        return df

    def upload_to_gcs(self, file_path, buffer):
        client = storage.Client()
        bucket = client.bucket(self.GCS_BUCKET_NAME)
        blob = bucket.blob(file_path)
        buffer.seek(0)
        blob.upload_from_file(
            buffer,
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        self.log(f"✅ Uploaded results to gs://{self.GCS_BUCKET_NAME}/{file_path}")

    

    def clean_df(self, df):
        df = df.astype(str).apply(lambda x: x.str.strip())
        df = df.replace(["", "nan", "NaT", "None", "NoneType"], pd.NA).fillna("Null")
        return df.drop_duplicates()

    def parse_data(self, value):
        if pd.isna(value):
            return None
        nums = re.findall(r"(\d+\.?\d*)", str(value))
        if not nums:
            return None
        if len(nums) == 2:
            return float(nums[0]), float(nums[1])
        try:
            return float(nums[0])
        except ValueError:
            return None

    def number_match(self, PDF_data, Excel_data):
        parsed_compare = self.parse_data(PDF_data)
        parsed_master = self.parse_data(Excel_data)
        if parsed_compare is None or parsed_master is None:
            return 0.0
        if isinstance(parsed_compare, tuple):
            PDF_low, PDF_high = parsed_compare
            Excel_val = parsed_master if not isinstance(parsed_master, tuple) else sum(parsed_master) / 2
            if PDF_low <= Excel_val <= PDF_high:
                return 100.0
            min_dist = min(abs(PDF_low - Excel_val), abs(PDF_high - Excel_val))
            if PDF_low == 0:
                return 0.0
            similarity = (1 - (min_dist / max(PDF_low, Excel_val))) * 100
            return round(max(0.0, similarity), 2)
        diff = abs(parsed_master - parsed_compare)
        if parsed_compare == 0:
            return 0.0
        similarity = (1 - (diff / max(parsed_master, parsed_compare))) * 100
        return round(max(0.0, similarity), 2)

    def retail_amount_match(self, PDF_amt, Excel_amt, PDF_factor, Excel_factor):
        try:
            v1_amt = float(str(PDF_amt).replace("$", "").replace(",", ""))
            v2_amt = float(str(Excel_amt).replace("$", "").replace(",", ""))
            v1_factor = float(str(PDF_factor).replace("$", "").replace(",", ""))
            v2_factor = float(str(Excel_factor).replace("$", "").replace(",", ""))
            if v1_factor == 0 or v2_factor == 0:
                return 0.0
            v1 = v1_amt / v1_factor
            v2 = v2_amt / v2_factor
        except Exception:
            return 0.0
        if v1 == v2:
            return 100.0
        diff = abs(v1 - v2)
        denominator = max(abs(v1), abs(v2), 1e-9)
        similarity = (1 - (diff / denominator)) * 100
        return round(max(0.0, similarity), 2)

    # -------------------------
    # SIMILARITY CLEANUP
    # -------------------------
    def clean_similarity_columns(self, df, columns):
        for col in columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # Only fill and round the columns that exist
        existing = [c for c in columns if c in df.columns]
        if existing:
            df[existing] = df[existing].fillna(0).round(2)
        return df

    # -------------------------
    # DEDUPLICATION & RANKING
    # -------------------------
    def deduplicate_exact_matches(self, df):
        if df is None or df.empty:
            return pd.DataFrame(columns=df.columns if df is not None else [])

        similarity_columns = [
            "Boost_Score", "Description_Similarity", "Ncat_Description_Similarity",
            "Brand_Description_Similarity", "Combined_Score", "LLM_Similarity",
            "Pack_Similarity", "Retail_Amt_Similarity", "Size_Similarity",
            "Brand_Word_Score", "Retail_Factor_Similarity",
        ]

        df = self.clean_similarity_columns(df, similarity_columns)

        # Identify rows having retail data
        has_retail = (
            df["PDF_Retail_Amt"].notna() & df["PDF_Retail_Factor"].notna() &
            df["Excel_Retail_Amt"].notna() & df["Excel_Retail_Factor"].notna()
            if all(col in df.columns for col in [
                "PDF_Retail_Amt", "PDF_Retail_Factor", "Excel_Retail_Amt", "Excel_Retail_Factor"
            ])
            else pd.Series(False, index=df.index)
        )

        df_with_retail = df[has_retail].copy()
        df_without_retail = df[~has_retail].copy()

        # Sort with retail priority
        if not df_with_retail.empty:
            df_with_retail["retail_priority"] = (df_with_retail["Retail_Amt_Similarity"] >= 99).astype(int)
            df_with_retail = (
                df_with_retail
                .sort_values(
                    by=["Excel_Description", "retail_priority", "LLM_Similarity", "Combined_Score"],
                    ascending=[True, False, False, False]
                )
                .drop(columns=["retail_priority"], errors="ignore")
            )

        # Sort without retail data
        if not df_without_retail.empty:
            df_without_retail = df_without_retail.sort_values(
                by=["Excel_Description", "LLM_Similarity", "Combined_Score"],
                ascending=[True, False, False]
            )

        # Combine and pick the top record per master description
        result = pd.concat([df_with_retail, df_without_retail], ignore_index=True)
        if "Excel_Description" in result.columns:
            result = result.groupby("Excel_Description", as_index=False).first()
        return result


    def rank_partial_matches(self, df):
        if df is None or df.empty:
            return pd.DataFrame(columns=df.columns if df is not None else [])

        similarity_columns = [
            "Boost_Score", "Description_Similarity", "Ncat_Description_Similarity",
            "Brand_Description_Similarity", "Combined_Score", "LLM_Similarity",
            "Pack_Similarity", "Retail_Amt_Similarity", "Size_Similarity",
            "Brand_Word_Score", "Retail_Factor_Similarity",
        ]

        df = self.clean_similarity_columns(df, similarity_columns)

        has_retail = (
            df["PDF_Retail_Amt"].notna() & df["PDF_Retail_Factor"].notna() &
            df["Excel_Retail_Amt"].notna() & df["Excel_Retail_Factor"].notna()
            if all(col in df.columns for col in [
                "PDF_Retail_Amt", "PDF_Retail_Factor", "Excel_Retail_Amt", "Excel_Retail_Factor"
            ])
            else pd.Series(False, index=df.index)
        )

        df_with_retail = df[has_retail].copy()
        df_without_retail = df[~has_retail].copy()

        if not df_with_retail.empty:
            df_with_retail["retail_priority"] = (df_with_retail["Retail_Amt_Similarity"] >= 99).astype(int)
            df_with_retail = (
                df_with_retail
                .sort_values(
                    by=["Excel_Description", "retail_priority", "LLM_Similarity", "Combined_Score"],
                    ascending=[True, False, False, False]
                )
                .drop(columns=["retail_priority"], errors="ignore")
            )

        if not df_without_retail.empty:
            df_without_retail = df_without_retail.sort_values(
                by=["Excel_Description", "LLM_Similarity", "Combined_Score"],
                ascending=[True, False, False]
            )

        # Combine and rank top 5 per master description
        result = pd.concat([df_with_retail, df_without_retail], ignore_index=True)
        if "Excel_Description" in result.columns:
            result["Match_Order"] = result.groupby("Excel_Description").cumcount() + 1
            result = result[result["Match_Order"] <= 5]
        return result

    # -------------------------
    # UTILS
    # -------------------------
    def normalize_score_to_0_1(self, val):
        """If val looks like percent (0-100) convert to 0-1, else pass through."""
        try:
            if pd.isna(val):
                return 0.0
            v = float(val)
            if v > 1.0:
                # assume percentage scaling
                return v / 100.0
            return v
        except Exception:
            try:
                # string like '0.85' etc
                return float(str(val).strip())
            except Exception:
                return 0.0

    # -------------------------
    # MAIN RUN
    # -------------------------
    def run(self):
        start_time = datetime.now()
        self.log("📥 Loading input data from GCS...")

        # Load LLM results & normalize columns
        llm_df = self.load_excel_from_gcs(self.INPUT_FILE_PATH)
        llm_df = self.normalize_columns(llm_df)

        # Convert placeholder nulls to proper NaN where appropriate (but keep strings safe)
        llm_df = llm_df.replace(["", "nan", "NaT", "None"], pd.NA).fillna("Null")

        # Make uppercase normalized description columns if present
        for col in ["PDF_Description", "Excel_Description"]:
            if col in llm_df.columns:
                llm_df[col] = llm_df[col].astype(str).str.strip().str.upper()

        # Load master reference & normalize
        excel_df = self.load_excel_from_gcs(self.EXCEL_FILE_PATH)
        excel_df = self.normalize_columns(excel_df)

        excel_df = self.clean_df(excel_df)
        if "Description" in excel_df.columns:
            excel_df["Description"] = excel_df["Description"].astype(str).str.strip().str.upper()

        self.log("🧮 Calculating numeric similarity metrics...")

        # Ensure columns exist and use get(...) when applying similarities
            # ==========================================
        # APPLY SIMILARITY CALCULATIONS
        # ==========================================
        self.log("📊 Applying similarity calculations...")

        def apply_similarity(row):
            return pd.Series(
                [
                    self.number_match(row.get("PDF_Size", None), row.get("Excel_Size", None)),
                    self.number_match(row.get("PDF_Pack", None), row.get("Excel_Pack", None)),
                    self.retail_amount_match(
                        row.get("PDF_Retail_Amt", 0),
                        row.get("Excel_Retail_Amt", 0),
                        row.get("PDF_Retail_Factor", 0),
                        row.get("Excel_Retail_Factor", 0),
                    ),
                    self.number_match(row.get("PDF_Retail_Factor", 0), row.get("Excel_Retail_Factor", 0)),
                ],
                index=[
                    "Size_Similarity",
                    "Pack_Similarity",
                    "Retail_Amt_Similarity",
                    "Retail_Factor_Similarity",
                ],
            )

        if not llm_df.empty:
            llm_df[
                ["Size_Similarity", "Pack_Similarity", "Retail_Amt_Similarity", "Retail_Factor_Similarity"]
            ] = llm_df.apply(apply_similarity, axis=1)
        else:
            self.log("⚠️ LLM dataframe is empty. Aborting run.")
            return

        # ==========================================
        # CLASSIFY MATCHES AND PARTIALS
        # ==========================================
        self.log("🔍 Classifying matched and partial records...")

        # Normalize numeric columns for safe comparisons
        numeric_candidates = [
            "LLM_Similarity",
            "Brand_Word_Score",
            "ncat_word_score",
            "Boost_Score",
            "Description_Similarity",
            "Combined_Score",
        ]
        for c in numeric_candidates:
            if c in llm_df.columns:
                llm_df[c] = pd.to_numeric(llm_df[c], errors="coerce").fillna(0)

        matched_rows, partial_rows, others = [], [], []

        for _, row in llm_df.iterrows():
            llm_sim = row.get("LLM_Similarity", 0)
            Brand_Word_Score = float(row.get("Brand_Word_Score", 0))
            ncat_sim = float(row.get("ncat_word_score", 0))
            size_score = float(row.get("Size_Similarity", 0))
            retail_amt_score = float(row.get("Retail_Amt_Similarity", 0))

            # Candidate filtering condition

            if (llm_sim >= 0.7) or (Brand_Word_Score >= 0.85 and ncat_sim >= 0.2):
                # Strict exact match criteria
                if (llm_sim >= 0.7) and (retail_amt_score == 100) and (size_score >= 80):
                    matched_rows.append(row)
                else:
                    partial_rows.append(row)
            else:
                others.append(row)


        # ==========================================
        # BUILD MATCHED AND PARTIAL DATAFRAMES
        # ==========================================
        matched_df = self.clean_df(pd.DataFrame(matched_rows))
        partial_df = self.clean_df(pd.DataFrame(partial_rows))

        # Deduplicate & rank
        matched_df = self.deduplicate_exact_matches(matched_df)
        partial_df = self.rank_partial_matches(partial_df)

        # Remove partial matches that already exist in exact matches
        if not matched_df.empty and not partial_df.empty and "Excel_Description" in partial_df.columns and "Excel_Description" in matched_df.columns:
            partial_df = partial_df[~partial_df["Excel_Description"].isin(matched_df["Excel_Description"])].reset_index(drop=True)

        # ==========================================
        # REORDER COLUMNS
        # ==========================================
        exact_match_columns_order = [
            'PDF_Description','PDF_Size','PDF_Pack','PDF_Retail_Factor','PDF_Retail_Amt','PDF_Page_Number',
            'Excel_Description','Excel_Size','Excel_Pack','Excel_Retail_Factor','Excel_Retail_Amt','Excel_nCat_Desc','Excel_BRAND',
            'Boost_Score','Description_Similarity','Ncat_Description_Similarity','Brand_Description_Similarity',
            'Combined_Score','LLM_Similarity','Brand_Word_Score','Size_Similarity','Pack_Similarity',
            'Retail_Amt_Similarity','Retail_Factor_Similarity'
        ]

        partial_match_columns_order = [
            'PDF_Description','PDF_Size','PDF_Pack','PDF_Retail_Factor','PDF_Retail_Amt','PDF_Page_Number',
            'Excel_Description','Excel_Size','Excel_Pack','Excel_Retail_Factor','Excel_Retail_Amt','Excel_nCat_Desc','Excel_BRAND',
            'Boost_Score','Description_Similarity','Ncat_Description_Similarity','Brand_Description_Similarity',
            'Combined_Score','LLM_Similarity','Brand_Word_Score','Size_Similarity','Pack_Similarity',
            'Retail_Amt_Similarity','Retail_Factor_Similarity','Match_Order'
        ]

        # Keep only existing columns (to avoid KeyErrors)
        exact_match_columns_order = [col for col in exact_match_columns_order if col in matched_df.columns]
        partial_match_columns_order = [col for col in partial_match_columns_order if col in partial_df.columns]

        # Apply column ordering
        if not matched_df.empty:
            matched_df = matched_df[exact_match_columns_order]
        if not partial_df.empty:
            partial_df = partial_df[partial_match_columns_order]


        # Build unmatched sets safely
        # PDF_unmatched: those PDF descriptions not included in matched or partial
        # Build unmatched sets safely
        # ==========================================
        # BUILD UNMATCHED PDF RECORDS SAFELY
        # ==========================================
        PDF_cols_prefixed = [
            'PDF_Description', 'PDF_Size', 'PDF_Pack',
            'PDF_Retail_Factor', 'PDF_Retail_Amt', 'PDF_Page_Number'
        ]

        # Ensure matched_df, partial_df, llm_df are all valid DataFrames
        matched_df = matched_df if matched_df is not None else pd.DataFrame()
        partial_df = partial_df if partial_df is not None else pd.DataFrame()
        llm_df = llm_df if llm_df is not None else pd.DataFrame()

        # Collect all matched PDF_Descriptions safely
        pdf_descs_matched = set()
        if not matched_df.empty and "PDF_Description" in matched_df.columns:
            pdf_descs_matched.update(matched_df["PDF_Description"].astype(str).tolist())
        if not partial_df.empty and "PDF_Description" in partial_df.columns:
            pdf_descs_matched.update(partial_df["PDF_Description"].astype(str).tolist())

        # Build unmatched set only if llm_df has the PDF_Description column
        if not llm_df.empty and "PDF_Description" in llm_df.columns:
            PDF_unmatched = llm_df[
                ~llm_df["PDF_Description"].astype(str).isin(pdf_descs_matched)
            ].copy()

            # Select only columns that exist
            available_cols = [col for col in PDF_cols_prefixed if col in PDF_unmatched.columns]
            PDF_unmatched = PDF_unmatched[available_cols].drop_duplicates(keep='first')

            # Rename to clean column names for final Excel output
            rename_map = {
                'PDF_Description': 'Description',
                'PDF_Size': 'Size',
                'PDF_Pack': 'Pack',
                'PDF_Retail_Factor': 'Retail_Factor',
                'PDF_Retail_Amt': 'Retail_Amt',
                'PDF_Page_Number': 'Page_Number'
            }
            PDF_unmatched.rename(columns=rename_map, inplace=True)
        else:
            PDF_unmatched = pd.DataFrame(
                columns=['Description', 'Size', 'Pack', 'Retail_Factor', 'Retail_Amt', 'Page_Number']
            )


        excel_descs_matched = set()
        if "Excel_Description" in matched_df.columns:
            excel_descs_matched.update(matched_df["Excel_Description"].astype(str).tolist())
        if "Excel_Description" in partial_df.columns:
            excel_descs_matched.update(partial_df["Excel_Description"].astype(str).tolist())

        if "Description" in excel_df.columns:
            Excel_unmatched = excel_df[~excel_df["Description"].astype(str).isin(excel_descs_matched)].copy()
        else:
            Excel_unmatched = excel_df.copy()

        self.log(f"✅ PDF_unmatched rows: {len(PDF_unmatched)}")
        self.log(f"✅ Excel_unmatched rows: {len(Excel_unmatched)}")
        self.log(f"Matched_df rows: {len(matched_df) if matched_df is not None else 0}")
        self.log(f"Partial_df rows: {len(partial_df) if partial_df is not None else 0}")

        # ==========================================
        # BUILD Match_Flag SHEET (Excel-based Classification)
        # ==========================================

        # Safely extract Excel descriptions from matched and partial data
        matched_Excel_desc = (
            set(matched_df["Excel_Description"].dropna().astype(str).tolist())
            if not matched_df.empty and "Excel_Description" in matched_df.columns
            else set()
        )

        partial_Excel_desc = (
            set(partial_df["Excel_Description"].dropna().astype(str).tolist())
            if not partial_df.empty and "Excel_Description" in partial_df.columns
            else set()
        )

        # Classification function
        def classify_Excel_row(row):
            desc = str(row.get("Description", "")).strip()
            if desc in matched_Excel_desc:
                return "Exact Match"
            elif desc in partial_Excel_desc:
                return "Partial Match"
            else:
                return "Unmatched"

        # Apply classification safely
        if not excel_df.empty and "Description" in excel_df.columns:
            excel_df["Match_Flag"] = excel_df.apply(classify_Excel_row, axis=1)
        else:
            excel_df["Match_Flag"] = "Unmatched"

        # Copy final DataFrame for output
        match_flag_df = excel_df.copy()


        # Write formatted Excel
        self.log("💾 Writing formatted Excel output...")
        output_buffer = io.BytesIO()
        with pd.ExcelWriter(output_buffer, engine="xlsxwriter") as writer:
            # Ensure we write something even if frames are empty
            matched_df.to_excel(writer, sheet_name="Exact_Matches", index=False)
            partial_df.to_excel(writer, sheet_name="Partial_Matches", index=False)
            PDF_unmatched.to_excel(writer, sheet_name="PDF_Unmatched", index=False)
            Excel_unmatched.to_excel(writer, sheet_name="Excel_Unmatched", index=False)
            match_flag_df.to_excel(writer, sheet_name="Match_Flag", index=False)

            workbook = writer.book
            fmt_compare = workbook.add_format({"bg_color": "#C6E2FC", "bold": True, "align": "center", "border": 1})
            fmt_master = workbook.add_format({"bg_color": "#C6FCC9", "bold": True, "align": "center", "border": 1})
            fmt_yellow = workbook.add_format({"bg_color": "#F5FD6A", "bold": True, "align": "center", "border": 1})
            key_formats = workbook.add_format({"bg_color": "#83CC75", "bold": True, "align": "center", "border": 2})
            fmt_flags = workbook.add_format({"bg_color": "#FF6B6B", "bold": True, "align": "center", "border": 1})

            PDF_keywords = ["PDF_"]
            Excel_keywords = ["Excel_"]
            key_Excel_cols = ["BRAND", "Description", "Pack", "Size", "Retail_Factor", "Retail_Amt", "nCat Desc"]
            Flags=["Match_Order","Match_Flag"]

            def color_headers(sheet_name, df):
                worksheet = writer.sheets[sheet_name]
                # Freeze top header row
                try:
                    worksheet.freeze_panes(1, 0)
                except Exception:
                    pass
                for col_num, col_name in enumerate(df.columns):
                    cname_upper = col_name.upper()
                    if sheet_name in ["Exact_Matches", "Partial_Matches"]:
                        if any(k in col_name for k in PDF_keywords):
                            fmt = fmt_compare
                        elif any(k in col_name for k in Excel_keywords):
                            fmt = fmt_master
                        elif any(k in col_name for k in Flags):
                            fmt = fmt_flags
                        else:
                            fmt = fmt_yellow
                    elif sheet_name == "PDF_Unmatched":
                        fmt = fmt_compare
                    elif sheet_name == "Excel_Unmatched":
                        fmt = key_formats if cname_upper in [c.upper() for c in key_Excel_cols] else fmt_master
                    elif sheet_name == "Match_Flag":
                      if any(k in col_name for k in Flags):
                        fmt = fmt_flags
                      else:
                        fmt = fmt_master
                    else:
                        fmt = fmt_yellow

                    # Write the header cell with format
                    try:
                        worksheet.write(0, col_num, col_name, fmt)
                    except Exception:
                        pass

                    # Compute column width (safely)
                    try:
                        col_width = max(df[col_name].astype(str).map(len).max(), len(col_name))
                        worksheet.set_column(col_num, col_num, min(col_width + 2, 60))
                    except Exception:
                        # fallback width
                        worksheet.set_column(col_num, col_num, 20)

            formatting_targets = [
                ("Exact_Matches", matched_df),
                ("Partial_Matches", partial_df),
                ("PDF_Unmatched", PDF_unmatched),
                ("Excel_Unmatched", Excel_unmatched),
                ("Match_Flag", match_flag_df),
            ]

            for sheet_name, df in formatting_targets:
                if df is not None and not df.empty:
                    color_headers(sheet_name, df)

        # Upload
        self.upload_to_gcs(self.OUTPUT_FILE_PATH, output_buffer)

        total_seconds = (datetime.now() - start_time).total_seconds()
        minutes, seconds = divmod(total_seconds, 60)
        self.log(f"✅ Final Similarity Pipeline completed in {int(minutes)}m {int(seconds)}s ({total_seconds:.2f}s)")

# -------------------------
# Example usage
# -------------------------# ===========================
# final_match_component.py
# ===========================
import os
import re
import io
import sys
import time
import json
import base64
import random
import hashlib
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import storage
# from sentence_transformers import SentenceTransformer  # keep if you use SBERT elsewhere
from rapidfuzz import fuzz
import google.genai as genai

# ===========================
# GENAI CONFIG (optional)
# ===========================
PROJECT_ID = "gc-proj-aiml-dev-01fd"
if not PROJECT_ID or PROJECT_ID == "[your-project-id]":
    PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT", ""))

LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
try:
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
except Exception:
    client = None  # optional; keep safe if genai not configured

# ===========================
# FINAL MATCH COMPONENT
# ===========================
class FinalMatchComponent:
    def __init__(self, bucket_name, pdf_input_path, master_file_path, pipeline_root_prefix="DPAC/6.pipelineRun",verbose: bool = True):
        self.GCS_BUCKET_NAME = bucket_name
        self.pdf_input_path = pdf_input_path
        self.master_file_path = master_file_path

        self.GCS_PIPELINE_ROOT_PREFIX = pipeline_root_prefix
        pdf_name_only = os.path.splitext(os.path.basename(self.pdf_input_path))[0]
        self.PDF_ROOT_PREFIX = f"{self.GCS_PIPELINE_ROOT_PREFIX}/{pdf_name_only}"

        self.INPUT_FILE_PATH = f"{self.PDF_ROOT_PREFIX}/LLM_Results.xlsx"
        self.OUTPUT_FILE_PATH = f"{self.PDF_ROOT_PREFIX}/Comparison_Results.xlsx"
        self.EXCEL_FILE_PATH = f"{self.GCS_PIPELINE_ROOT_PREFIX}/{self.master_file_path}"
        self.verbose=verbose

        self.PIPELINE_START = datetime.now()

    # -------------------------
    # LOGGING
    # -------------------------
    def log(self, msg: str):
      if self.verbose:  # only log when verbose=True
        elapsed = datetime.now() - self.PIPELINE_START
        m, s = divmod(elapsed.total_seconds(), 60)
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} | ⏱ {int(m)}m {int(s)}s | {msg}")
    
    def normalize_columns(self, df):
        """Replace spaces with underscores and strip column names for consistent access."""
        df = df.copy()
        df.columns = (
            df.columns.astype(str)
            .str.strip()
            .str.replace(" ", "_")
            .str.replace(r"[^\w_]", "", regex=True)
        )
        return df
    # -------------------------
    # GCS HELPERS
    # -------------------------
    def load_excel_from_gcs(self, file_path, usecols=None):
        client = storage.Client()
        bucket = client.bucket(self.GCS_BUCKET_NAME)
        blob = bucket.blob(file_path)
        content = blob.download_as_bytes()
        df = pd.read_excel(io.BytesIO(content), usecols=usecols)
        df.columns = df.columns.str.strip()
        self.log(f"Loaded '{file_path}' from GCS. Shape: {df.shape}")
        return df

    def upload_to_gcs(self, file_path, buffer):
        client = storage.Client()
        bucket = client.bucket(self.GCS_BUCKET_NAME)
        blob = bucket.blob(file_path)
        buffer.seek(0)
        blob.upload_from_file(
            buffer,
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        self.log(f"✅ Uploaded results to gs://{self.GCS_BUCKET_NAME}/{file_path}")

    

    def clean_df(self, df):
        df = df.astype(str).apply(lambda x: x.str.strip())
        df = df.replace(["", "nan", "NaT", "None", "NoneType"], pd.NA).fillna("Null")
        return df.drop_duplicates()

    def parse_data(self, value):
        if pd.isna(value):
            return None
        nums = re.findall(r"(\d+\.?\d*)", str(value))
        if not nums:
            return None
        if len(nums) == 2:
            return float(nums[0]), float(nums[1])
        try:
            return float(nums[0])
        except ValueError:
            return None

    def number_match(self, PDF_data, Excel_data):
        parsed_compare = self.parse_data(PDF_data)
        parsed_master = self.parse_data(Excel_data)
        if parsed_compare is None or parsed_master is None:
            return 0.0
        if isinstance(parsed_compare, tuple):
            PDF_low, PDF_high = parsed_compare
            Excel_val = parsed_master if not isinstance(parsed_master, tuple) else sum(parsed_master) / 2
            if PDF_low <= Excel_val <= PDF_high:
                return 100.0
            min_dist = min(abs(PDF_low - Excel_val), abs(PDF_high - Excel_val))
            if PDF_low == 0:
                return 0.0
            similarity = (1 - (min_dist / max(PDF_low, Excel_val))) * 100
            return round(max(0.0, similarity), 2)
        diff = abs(parsed_master - parsed_compare)
        if parsed_compare == 0:
            return 0.0
        similarity = (1 - (diff / max(parsed_master, parsed_compare))) * 100
        return round(max(0.0, similarity), 2)

    def retail_amount_match(self, PDF_amt, Excel_amt, PDF_factor, Excel_factor):
        try:
            v1_amt = float(str(PDF_amt).replace("$", "").replace(",", ""))
            v2_amt = float(str(Excel_amt).replace("$", "").replace(",", ""))
            v1_factor = float(str(PDF_factor).replace("$", "").replace(",", ""))
            v2_factor = float(str(Excel_factor).replace("$", "").replace(",", ""))
            if v1_factor == 0 or v2_factor == 0:
                return 0.0
            v1 = v1_amt / v1_factor
            v2 = v2_amt / v2_factor
        except Exception:
            return 0.0
        if v1 == v2:
            return 100.0
        diff = abs(v1 - v2)
        denominator = max(abs(v1), abs(v2), 1e-9)
        similarity = (1 - (diff / denominator)) * 100
        return round(max(0.0, similarity), 2)

    # -------------------------
    # SIMILARITY CLEANUP
    # -------------------------
    def clean_similarity_columns(self, df, columns):
        for col in columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # Only fill and round the columns that exist
        existing = [c for c in columns if c in df.columns]
        if existing:
            df[existing] = df[existing].fillna(0).round(2)
        return df

    # -------------------------
    # DEDUPLICATION & RANKING
    # -------------------------
    def deduplicate_exact_matches(self, df):
        if df is None or df.empty:
            return pd.DataFrame(columns=df.columns if df is not None else [])

        similarity_columns = [
            "Boost_Score", "Description_Similarity", "Ncat_Description_Similarity",
            "Brand_Description_Similarity", "Combined_Score", "LLM_Similarity",
            "Pack_Similarity", "Retail_Amt_Similarity", "Size_Similarity",
            "Brand_Word_Score", "Retail_Factor_Similarity",
        ]

        df = self.clean_similarity_columns(df, similarity_columns)

        # Identify rows having retail data
        has_retail = (
            df["PDF_Retail_Amt"].notna() & df["PDF_Retail_Factor"].notna() &
            df["Excel_Retail_Amt"].notna() & df["Excel_Retail_Factor"].notna()
            if all(col in df.columns for col in [
                "PDF_Retail_Amt", "PDF_Retail_Factor", "Excel_Retail_Amt", "Excel_Retail_Factor"
            ])
            else pd.Series(False, index=df.index)
        )

        df_with_retail = df[has_retail].copy()
        df_without_retail = df[~has_retail].copy()

        # Sort with retail priority
        if not df_with_retail.empty:
            df_with_retail["retail_priority"] = (df_with_retail["Retail_Amt_Similarity"] >= 99).astype(int)
            df_with_retail = (
                df_with_retail
                .sort_values(
                    by=["Excel_Description", "retail_priority", "LLM_Similarity", "Combined_Score"],
                    ascending=[True, False, False, False]
                )
                .drop(columns=["retail_priority"], errors="ignore")
            )

        # Sort without retail data
        if not df_without_retail.empty:
            df_without_retail = df_without_retail.sort_values(
                by=["Excel_Description", "LLM_Similarity", "Combined_Score"],
                ascending=[True, False, False]
            )

        # Combine and pick the top record per master description
        result = pd.concat([df_with_retail, df_without_retail], ignore_index=True)
        if "Excel_Description" in result.columns:
            result = result.groupby("Excel_Description", as_index=False).first()
        return result


    def rank_partial_matches(self, df):
        if df is None or df.empty:
            return pd.DataFrame(columns=df.columns if df is not None else [])

        similarity_columns = [
            "Boost_Score", "Description_Similarity", "Ncat_Description_Similarity",
            "Brand_Description_Similarity", "Combined_Score", "LLM_Similarity",
            "Pack_Similarity", "Retail_Amt_Similarity", "Size_Similarity",
            "Brand_Word_Score", "Retail_Factor_Similarity",
        ]

        df = self.clean_similarity_columns(df, similarity_columns)

        has_retail = (
            df["PDF_Retail_Amt"].notna() & df["PDF_Retail_Factor"].notna() &
            df["Excel_Retail_Amt"].notna() & df["Excel_Retail_Factor"].notna()
            if all(col in df.columns for col in [
                "PDF_Retail_Amt", "PDF_Retail_Factor", "Excel_Retail_Amt", "Excel_Retail_Factor"
            ])
            else pd.Series(False, index=df.index)
        )

        df_with_retail = df[has_retail].copy()
        df_without_retail = df[~has_retail].copy()

        if not df_with_retail.empty:
            df_with_retail["retail_priority"] = (df_with_retail["Retail_Amt_Similarity"] >= 99).astype(int)
            df_with_retail = (
                df_with_retail
                .sort_values(
                    by=["Excel_Description", "retail_priority", "LLM_Similarity", "Combined_Score"],
                    ascending=[True, False, False, False]
                )
                .drop(columns=["retail_priority"], errors="ignore")
            )

        if not df_without_retail.empty:
            df_without_retail = df_without_retail.sort_values(
                by=["Excel_Description", "LLM_Similarity", "Combined_Score"],
                ascending=[True, False, False]
            )

        # Combine and rank top 5 per master description
        result = pd.concat([df_with_retail, df_without_retail], ignore_index=True)
        if "Excel_Description" in result.columns:
            result["Match_Order"] = result.groupby("Excel_Description").cumcount() + 1
            result = result[result["Match_Order"] <= 5]
        return result

    # -------------------------
    # UTILS
    # -------------------------
    def normalize_score_to_0_1(self, val):
        """If val looks like percent (0-100) convert to 0-1, else pass through."""
        try:
            if pd.isna(val):
                return 0.0
            v = float(val)
            if v > 1.0:
                # assume percentage scaling
                return v / 100.0
            return v
        except Exception:
            try:
                # string like '0.85' etc
                return float(str(val).strip())
            except Exception:
                return 0.0

    # -------------------------
    # MAIN RUN
    # -------------------------
    def run(self):
        start_time = datetime.now()
        self.log("📥 Loading input data from GCS...")

        # Load LLM results & normalize columns
        llm_df = self.load_excel_from_gcs(self.INPUT_FILE_PATH)
        llm_df = self.normalize_columns(llm_df)

        # Convert placeholder nulls to proper NaN where appropriate (but keep strings safe)
        llm_df = llm_df.replace(["", "nan", "NaT", "None"], pd.NA).fillna("Null")

        # Make uppercase normalized description columns if present
        for col in ["PDF_Description", "Excel_Description"]:
            if col in llm_df.columns:
                llm_df[col] = llm_df[col].astype(str).str.strip().str.upper()

        # Load master reference & normalize
        excel_df = self.load_excel_from_gcs(self.EXCEL_FILE_PATH)
        excel_df = self.normalize_columns(excel_df)

        excel_df = self.clean_df(excel_df)
        if "Description" in excel_df.columns:
            excel_df["Description"] = excel_df["Description"].astype(str).str.strip().str.upper()

        self.log("🧮 Calculating numeric similarity metrics...")

        # Ensure columns exist and use get(...) when applying similarities
            # ==========================================
        # APPLY SIMILARITY CALCULATIONS
        # ==========================================
        self.log("📊 Applying similarity calculations...")

        def apply_similarity(row):
            return pd.Series(
                [
                    self.number_match(row.get("PDF_Size", None), row.get("Excel_Size", None)),
                    self.number_match(row.get("PDF_Pack", None), row.get("Excel_Pack", None)),
                    self.retail_amount_match(
                        row.get("PDF_Retail_Amt", 0),
                        row.get("Excel_Retail_Amt", 0),
                        row.get("PDF_Retail_Factor", 0),
                        row.get("Excel_Retail_Factor", 0),
                    ),
                    self.number_match(row.get("PDF_Retail_Factor", 0), row.get("Excel_Retail_Factor", 0)),
                ],
                index=[
                    "Size_Similarity",
                    "Pack_Similarity",
                    "Retail_Amt_Similarity",
                    "Retail_Factor_Similarity",
                ],
            )

        if not llm_df.empty:
            llm_df[
                ["Size_Similarity", "Pack_Similarity", "Retail_Amt_Similarity", "Retail_Factor_Similarity"]
            ] = llm_df.apply(apply_similarity, axis=1)
        else:
            self.log("⚠️ LLM dataframe is empty. Aborting run.")
            return

        # ==========================================
        # CLASSIFY MATCHES AND PARTIALS
        # ==========================================
        self.log("🔍 Classifying matched and partial records...")

        # Normalize numeric columns for safe comparisons
        numeric_candidates = [
            "LLM_Similarity",
            "Brand_Word_Score",
            "ncat_word_score",
            "Boost_Score",
            "Description_Similarity",
            "Combined_Score",
        ]
        for c in numeric_candidates:
            if c in llm_df.columns:
                llm_df[c] = pd.to_numeric(llm_df[c], errors="coerce").fillna(0)

        matched_rows, partial_rows, others = [], [], []

        for _, row in llm_df.iterrows():
            llm_sim = row.get("LLM_Similarity", 0)
            Brand_Word_Score = float(row.get("Brand_Word_Score", 0))
            ncat_sim = float(row.get("ncat_word_score", 0))
            size_score = float(row.get("Size_Similarity", 0))
            retail_amt_score = float(row.get("Retail_Amt_Similarity", 0))

            # Candidate filtering condition

            if (llm_sim >= 0.7) or (Brand_Word_Score >= 0.85 and ncat_sim >= 0.2):
                # Strict exact match criteria
                if (llm_sim >= 0.7) and (retail_amt_score == 100) and (size_score >= 80):
                    matched_rows.append(row)
                else:
                    partial_rows.append(row)
            else:
                others.append(row)


        # ==========================================
        # BUILD MATCHED AND PARTIAL DATAFRAMES
        # ==========================================
        matched_df = self.clean_df(pd.DataFrame(matched_rows))
        partial_df = self.clean_df(pd.DataFrame(partial_rows))

        # Deduplicate & rank
        matched_df = self.deduplicate_exact_matches(matched_df)
        partial_df = self.rank_partial_matches(partial_df)

        # Remove partial matches that already exist in exact matches
        if not matched_df.empty and not partial_df.empty and "Excel_Description" in partial_df.columns and "Excel_Description" in matched_df.columns:
            partial_df = partial_df[~partial_df["Excel_Description"].isin(matched_df["Excel_Description"])].reset_index(drop=True)

        # ==========================================
        # REORDER COLUMNS
        # ==========================================
        exact_match_columns_order = [
            'PDF_Description','PDF_Size','PDF_Pack','PDF_Retail_Factor','PDF_Retail_Amt','PDF_Page_Number',
            'Excel_Description','Excel_Size','Excel_Pack','Excel_Retail_Factor','Excel_Retail_Amt','Excel_nCat_Desc','Excel_BRAND',
            'Boost_Score','Description_Similarity','Ncat_Description_Similarity','Brand_Description_Similarity',
            'Combined_Score','LLM_Similarity','Brand_Word_Score','Size_Similarity','Pack_Similarity',
            'Retail_Amt_Similarity','Retail_Factor_Similarity'
        ]

        partial_match_columns_order = [
            'PDF_Description','PDF_Size','PDF_Pack','PDF_Retail_Factor','PDF_Retail_Amt','PDF_Page_Number',
            'Excel_Description','Excel_Size','Excel_Pack','Excel_Retail_Factor','Excel_Retail_Amt','Excel_nCat_Desc','Excel_BRAND',
            'Boost_Score','Description_Similarity','Ncat_Description_Similarity','Brand_Description_Similarity',
            'Combined_Score','LLM_Similarity','Brand_Word_Score','Size_Similarity','Pack_Similarity',
            'Retail_Amt_Similarity','Retail_Factor_Similarity','Match_Order'
        ]

        # Keep only existing columns (to avoid KeyErrors)
        exact_match_columns_order = [col for col in exact_match_columns_order if col in matched_df.columns]
        partial_match_columns_order = [col for col in partial_match_columns_order if col in partial_df.columns]

        # Apply column ordering
        if not matched_df.empty:
            matched_df = matched_df[exact_match_columns_order]
        if not partial_df.empty:
            partial_df = partial_df[partial_match_columns_order]


        # Build unmatched sets safely
        # PDF_unmatched: those PDF descriptions not included in matched or partial
        # Build unmatched sets safely
        # ==========================================
        # BUILD UNMATCHED PDF RECORDS SAFELY
        # ==========================================
        PDF_cols_prefixed = [
            'PDF_Description', 'PDF_Size', 'PDF_Pack',
            'PDF_Retail_Factor', 'PDF_Retail_Amt', 'PDF_Page_Number'
        ]

        # Ensure matched_df, partial_df, llm_df are all valid DataFrames
        matched_df = matched_df if matched_df is not None else pd.DataFrame()
        partial_df = partial_df if partial_df is not None else pd.DataFrame()
        llm_df = llm_df if llm_df is not None else pd.DataFrame()

        # Collect all matched PDF_Descriptions safely
        pdf_descs_matched = set()
        if not matched_df.empty and "PDF_Description" in matched_df.columns:
            pdf_descs_matched.update(matched_df["PDF_Description"].astype(str).tolist())
        if not partial_df.empty and "PDF_Description" in partial_df.columns:
            pdf_descs_matched.update(partial_df["PDF_Description"].astype(str).tolist())

        # Build unmatched set only if llm_df has the PDF_Description column
        if not llm_df.empty and "PDF_Description" in llm_df.columns:
            PDF_unmatched = llm_df[
                ~llm_df["PDF_Description"].astype(str).isin(pdf_descs_matched)
            ].copy()

            # Select only columns that exist
            available_cols = [col for col in PDF_cols_prefixed if col in PDF_unmatched.columns]
            PDF_unmatched = PDF_unmatched[available_cols].drop_duplicates(keep='first')

            # Rename to clean column names for final Excel output
            rename_map = {
                'PDF_Description': 'Description',
                'PDF_Size': 'Size',
                'PDF_Pack': 'Pack',
                'PDF_Retail_Factor': 'Retail_Factor',
                'PDF_Retail_Amt': 'Retail_Amt',
                'PDF_Page_Number': 'Page_Number'
            }
            PDF_unmatched.rename(columns=rename_map, inplace=True)
        else:
            PDF_unmatched = pd.DataFrame(
                columns=['Description', 'Size', 'Pack', 'Retail_Factor', 'Retail_Amt', 'Page_Number']
            )


        excel_descs_matched = set()
        if "Excel_Description" in matched_df.columns:
            excel_descs_matched.update(matched_df["Excel_Description"].astype(str).tolist())
        if "Excel_Description" in partial_df.columns:
            excel_descs_matched.update(partial_df["Excel_Description"].astype(str).tolist())

        if "Description" in excel_df.columns:
            Excel_unmatched = excel_df[~excel_df["Description"].astype(str).isin(excel_descs_matched)].copy()
        else:
            Excel_unmatched = excel_df.copy()

        self.log(f"✅ PDF_unmatched rows: {len(PDF_unmatched)}")
        self.log(f"✅ Excel_unmatched rows: {len(Excel_unmatched)}")
        self.log(f"Matched_df rows: {len(matched_df) if matched_df is not None else 0}")
        self.log(f"Partial_df rows: {len(partial_df) if partial_df is not None else 0}")

        # ==========================================
        # BUILD Match_Flag SHEET (Excel-based Classification)
        # ==========================================

        # Safely extract Excel descriptions from matched and partial data
        matched_Excel_desc = (
            set(matched_df["Excel_Description"].dropna().astype(str).tolist())
            if not matched_df.empty and "Excel_Description" in matched_df.columns
            else set()
        )

        partial_Excel_desc = (
            set(partial_df["Excel_Description"].dropna().astype(str).tolist())
            if not partial_df.empty and "Excel_Description" in partial_df.columns
            else set()
        )

        # Classification function
        def classify_Excel_row(row):
            desc = str(row.get("Description", "")).strip()
            if desc in matched_Excel_desc:
                return "Exact Match"
            elif desc in partial_Excel_desc:
                return "Partial Match"
            else:
                return "Unmatched"

        # Apply classification safely
        if not excel_df.empty and "Description" in excel_df.columns:
            excel_df["Match_Flag"] = excel_df.apply(classify_Excel_row, axis=1)
        else:
            excel_df["Match_Flag"] = "Unmatched"

        # Copy final DataFrame for output
        match_flag_df = excel_df.copy()


        # Write formatted Excel
        self.log("💾 Writing formatted Excel output...")
        output_buffer = io.BytesIO()
        with pd.ExcelWriter(output_buffer, engine="xlsxwriter") as writer:
            # Ensure we write something even if frames are empty
            matched_df.to_excel(writer, sheet_name="Exact_Matches", index=False)
            partial_df.to_excel(writer, sheet_name="Partial_Matches", index=False)
            PDF_unmatched.to_excel(writer, sheet_name="PDF_Unmatched", index=False)
            Excel_unmatched.to_excel(writer, sheet_name="Excel_Unmatched", index=False)
            match_flag_df.to_excel(writer, sheet_name="Match_Flag", index=False)

            workbook = writer.book
            fmt_compare = workbook.add_format({"bg_color": "#C6E2FC", "bold": True, "align": "center", "border": 1})
            fmt_master = workbook.add_format({"bg_color": "#C6FCC9", "bold": True, "align": "center", "border": 1})
            fmt_yellow = workbook.add_format({"bg_color": "#F5FD6A", "bold": True, "align": "center", "border": 1})
            key_formats = workbook.add_format({"bg_color": "#83CC75", "bold": True, "align": "center", "border": 2})
            fmt_flags = workbook.add_format({"bg_color": "#FF6B6B", "bold": True, "align": "center", "border": 1})

            PDF_keywords = ["PDF_"]
            Excel_keywords = ["Excel_"]
            key_Excel_cols = ["BRAND", "Description", "Pack", "Size", "Retail_Factor", "Retail_Amt", "nCat_Desc"]
            Flags=["Match_Order","Match_Flag"]

            def color_headers(sheet_name, df):
                worksheet = writer.sheets[sheet_name]
                # Freeze top header row
                try:
                    worksheet.freeze_panes(1, 0)
                except Exception:
                    pass
                for col_num, col_name in enumerate(df.columns):
                    cname_upper = col_name.upper()
                    if sheet_name in ["Exact_Matches", "Partial_Matches"]:
                        if any(k in col_name for k in PDF_keywords):
                            fmt = fmt_compare
                        elif any(k in col_name for k in Excel_keywords):
                            fmt = fmt_master
                        elif any(k in col_name for k in Flags):
                            fmt = fmt_flags
                        else:
                            fmt = fmt_yellow
                    elif sheet_name == "PDF_Unmatched":
                        fmt = fmt_compare
                    elif sheet_name == "Excel_Unmatched":
                        fmt = key_formats if cname_upper in [c.upper() for c in key_Excel_cols] else fmt_master
                    elif sheet_name == "Match_Flag":
                        if any(flag in col_name for flag in Flags):
                            fmt = fmt_flags
                        else:
                            # Apply key format if column name matches any key column (case-insensitive)
                            if col_name.upper() in (c.upper() for c in key_Excel_cols):
                                fmt = key_formats
                            else:
                                fmt = fmt_master

                    else:
                        fmt = fmt_yellow

                    # Write the header cell with format
                    try:
                        worksheet.write(0, col_num, col_name, fmt)
                    except Exception:
                        pass

                    # Compute column width (safely)
                    try:
                        col_width = max(df[col_name].astype(str).map(len).max(), len(col_name))
                        worksheet.set_column(col_num, col_num, min(col_width + 2, 60))
                    except Exception:
                        # fallback width
                        worksheet.set_column(col_num, col_num, 20)

            formatting_targets = [
                ("Exact_Matches", matched_df),
                ("Partial_Matches", partial_df),
                ("PDF_Unmatched", PDF_unmatched),
                ("Excel_Unmatched", Excel_unmatched),
                ("Match_Flag", match_flag_df),
            ]

            for sheet_name, df in formatting_targets:
                if df is not None and not df.empty:
                    color_headers(sheet_name, df)

        # Upload
        self.upload_to_gcs(self.OUTPUT_FILE_PATH, output_buffer)

        total_seconds = (datetime.now() - start_time).total_seconds()
        minutes, seconds = divmod(total_seconds, 60)
        self.log(f"✅ Final Similarity Pipeline completed in {int(minutes)}m {int(seconds)}s ({total_seconds:.2f}s)")

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Replace these with your actual bucket & file paths
    bucket = os.environ.get("GCS_BUCKET_NAME", "gc-us-gcs-aiml-dev")
    pdf_input = os.environ.get("PDF_INPUT_PATH", "NEW_PW/my_pdf.pdf")  # used to create PDF-root prefix
    master_file = os.environ.get("MASTER_FILE_PATH", "MasterFiles/Master.xlsx")

    component = FinalMatchComponent(bucket_name=bucket, pdf_input_path=pdf_input, master_file_path=master_file)
    component.run()

if __name__ == "__main__":
    # Replace these with your actual bucket & file paths
    bucket = os.environ.get("GCS_BUCKET_NAME", "gc-us-gcs-aiml-dev")
    pdf_input = os.environ.get("PDF_INPUT_PATH", "NEW_PW/my_pdf.pdf")  # used to create PDF-root prefix
    master_file = os.environ.get("MASTER_FILE_PATH", "MasterFiles/Master.xlsx")

    component = FinalMatchComponent(bucket_name=bucket, pdf_input_path=pdf_input, master_file_path=master_file)
    component.run()
