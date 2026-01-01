# ========================== Imports ==========================
import os
import io
import time
import json
import csv
import uuid
import logging
import numpy as np

from PIL import Image, ImageOps
from pathlib import Path
from pdf2image import convert_from_bytes
from datetime import datetime
from scipy.stats import entropy
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from google.cloud import storage
from ultralytics import YOLO

# ========================== ENV SETUP ==========================
if os.system("which pdfinfo > /dev/null 2>&1") != 0:
    print("🔧 Installing poppler-utils...")
    os.system("apt-get install -y poppler-utils")

os.environ["ULTRALYTICS_VERBOSE"] = "False"
os.environ["ULTRALYTICS_HIDE_CONFIG_WARNINGS"] = "True"

logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("engine").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)


# ========================== PIPELINE CLASS ==========================
class PDFProcessingPipeline:

    def __init__(self, gcs_bucket_name: str, input_pdf: str,
                 pipeline_root_prefix: str = "DPAC/8.TestRun"):

        self.GCS_BUCKET_NAME = gcs_bucket_name
        self.INPUT_PDF = input_pdf
        self.GCS_PIPELINE_ROOT_PREFIX = pipeline_root_prefix

        self.pdf_name_only = os.path.splitext(os.path.basename(self.INPUT_PDF))[0]
        self.PDF_ROOT_PREFIX = f"{self.GCS_PIPELINE_ROOT_PREFIX}/{self.pdf_name_only}"
        self.PDFS_INPUT_PREFIX = f"{self.GCS_PIPELINE_ROOT_PREFIX}/{self.INPUT_PDF}"
        self.OUTPUT_IMAGES_PREFIX = f"{self.PDF_ROOT_PREFIX}/converted_images"
        self.TEST_IMAGES_PREFIX = self.OUTPUT_IMAGES_PREFIX
        self.PREDICTION_RESULTS_PREFIX = f"{self.PDF_ROOT_PREFIX}/Full_Model_predicted_results"
        self.CROPPED_OBJECTS_PREFIX = f"{self.PDF_ROOT_PREFIX}/Cropped_Objects"
        self.CLEANED_IMAGES_PREFIX = f"{self.PDF_ROOT_PREFIX}/cleaned_images_new"
        self.TRAINED_MODEL_FILENAME = f"{self.GCS_PIPELINE_ROOT_PREFIX}/PW_Final_model.pt"

        # Params
        self.USE_GRAYSCALE = True
        self.IMG_SIZE = 1280
        self.LINE_WIDTH = 4
        self.DPI = 200
        self.JPEG_QUALITY = 95
        self.MAX_WORKERS = os.cpu_count()
        self.MAX_WORKERS_NOISE = 10
        self.ENTROPY_THRESHOLD = 1.7

        # Local paths
        self.LOCAL_MODEL_PATH = "/tmp/yolo_model.pt"
        self.LOCAL_IMAGE_DOWNLOAD_DIR = "/tmp/input_images_for_cropping"
        self.LOCAL_CROP_DIR = "/tmp/cropped_images_output"

        os.makedirs(self.LOCAL_IMAGE_DOWNLOAD_DIR, exist_ok=True)
        os.makedirs(self.LOCAL_CROP_DIR, exist_ok=True)

        self.PIPELINE_START = datetime.now()

    # ========================== Utilities ==========================
    def log(self, msg: str):
        elapsed = datetime.now() - self.PIPELINE_START
        m, s = divmod(elapsed.total_seconds(), 60)
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} | ⏱ {int(m)}m {int(s)}s | {msg}")

    def init_gcs_client(self):
        return storage.Client()

    def verify_bucket(self):
        bucket = self.init_gcs_client().bucket(self.GCS_BUCKET_NAME)
        if not bucket.exists():
            raise ValueError(f"Bucket '{self.GCS_BUCKET_NAME}' does not exist.")
        self.log(f"✅ Bucket '{self.GCS_BUCKET_NAME}' verified.")
        return bucket

    def download_blob(self, source_blob_name, dest_file):
        self.init_gcs_client().bucket(self.GCS_BUCKET_NAME) \
            .blob(source_blob_name).download_to_filename(dest_file)

    def upload_blob(self, local_file, dest_blob_name):
        self.init_gcs_client().bucket(self.GCS_BUCKET_NAME) \
            .blob(dest_blob_name).upload_from_filename(local_file)

    def list_pdfs(self, prefix):
        bucket = self.init_gcs_client().bucket(self.GCS_BUCKET_NAME)
        return [b.name for b in bucket.list_blobs(prefix=prefix)
                if b.name.lower().endswith(".pdf")]

    # ========================== Step 1: PDF → Images ==========================
    def convert_pdf_page_to_jpg(self, pdf_bytes, pdf_basename, output_prefix, page_number):
        bucket = self.init_gcs_client().bucket(self.GCS_BUCKET_NAME)
        try:
            images = convert_from_bytes(
                pdf_bytes, dpi=self.DPI, fmt="jpeg",
                first_page=page_number, last_page=page_number
            )
            img = images[0]
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=self.JPEG_QUALITY)
            buf.seek(0)
            filename = f"{pdf_basename}_page_{page_number}.jpg"
            bucket.blob(f"{output_prefix}/{filename}") \
                .upload_from_file(buf, content_type="image/jpeg")
            return 1
        except Exception as e:
            self.log(f"❌ Page {page_number} failed: {e}")
            return 0

    def process_pdf(self, pdf_blob_name, output_prefix):
        bucket = self.init_gcs_client().bucket(self.GCS_BUCKET_NAME)
        pdf_bytes = bucket.blob(pdf_blob_name).download_as_bytes()
        pdf_basename = os.path.splitext(os.path.basename(pdf_blob_name))[0]

        pages = convert_from_bytes(pdf_bytes, dpi=72)
        total_pages = len(pages)
        del pages

        with ProcessPoolExecutor(max_workers=min(self.MAX_WORKERS, total_pages)) as executor:
            futures = [
                executor.submit(
                    self.convert_pdf_page_to_jpg,
                    pdf_bytes, pdf_basename, output_prefix, i + 1
                )
                for i in range(total_pages)
            ]
            return sum(f.result() for f in futures)

    # ========================== Step 2: YOLO Inference ==========================
    def run_yolo_inference(self):
        self.download_blob(self.TRAINED_MODEL_FILENAME, self.LOCAL_MODEL_PATH)
        model = YOLO(self.LOCAL_MODEL_PATH)
        bucket = self.init_gcs_client().bucket(self.GCS_BUCKET_NAME)

        for blob in bucket.list_blobs(prefix=self.TEST_IMAGES_PREFIX):
            if not blob.name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            local_input = f"/tmp/{os.path.basename(blob.name)}"
            local_output = f"/tmp/out_{os.path.basename(blob.name)}"
            blob.download_to_filename(local_input)

            img = Image.open(local_input).convert("RGB")
            if self.USE_GRAYSCALE:
                img = ImageOps.grayscale(img).convert("RGB")

            results = model.predict(img, imgsz=self.IMG_SIZE, conf=0.4, verbose=False)
            annotated = results[0].plot(line_width=self.LINE_WIDTH)

            Image.fromarray(annotated).save(local_output, format="JPEG", quality=95)
            self.upload_blob(local_output, f"{self.PREDICTION_RESULTS_PREFIX}/{os.path.basename(blob.name)}")

    # ========================== Step 3: Object Cropping ==========================
    def process_image_crop(self, image_path, model, writer):
        img_pil = Image.open(image_path).convert("RGB")
        img = np.array(img_pil)
        h, w = img.shape[:2]

        results = model.predict(img, conf=0.4, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().tolist()

        cropped_count = 0

        for idx, (x1, y1, x2, y2) in enumerate(boxes):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                writer.writerow([os.path.basename(image_path), idx + 1, x1, y1, x2, y2, "Invalid box"])
                continue

            crop = img[y1:y2, x1:x2]
            crop_name = f"{Path(image_path).stem}_crop_{idx+1}_{uuid.uuid4().hex[:5]}.jpg"
            crop_path = os.path.join(self.LOCAL_CROP_DIR, crop_name)

            Image.fromarray(crop).save(crop_path, format="JPEG", quality=95)
            self.upload_blob(crop_path, f"{self.CROPPED_OBJECTS_PREFIX}/{crop_name}")
            cropped_count += 1

        return cropped_count

    def run_object_cropping(self):
        self.download_blob(self.TRAINED_MODEL_FILENAME, self.LOCAL_MODEL_PATH)
        model = YOLO(self.LOCAL_MODEL_PATH)

        bucket = self.init_gcs_client().bucket(self.GCS_BUCKET_NAME)
        images = [b for b in bucket.list_blobs(prefix=self.TEST_IMAGES_PREFIX)
                  if b.name.lower().endswith(".jpg")]

        skipped_log = os.path.join(self.LOCAL_CROP_DIR, "skipped_detections.csv")
        with open(skipped_log, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Image", "Idx", "x1", "y1", "x2", "y2", "Reason"])

            for blob in images:
                local = f"/tmp/{os.path.basename(blob.name)}"
                blob.download_to_filename(local)
                self.process_image_crop(local, model, writer)

    # ========================== Step 4: Noise Filtering ==========================
    def calculate_entropy(self, image):
        gray = image.convert("L")
        hist, _ = np.histogram(np.array(gray).flatten(), bins=256, density=True)
        hist = hist[hist > 0]
        return entropy(hist)

    def run_image_cleaning(self):
        bucket = self.init_gcs_client().bucket(self.GCS_BUCKET_NAME)
        blobs = list(bucket.list_blobs(prefix=self.CROPPED_OBJECTS_PREFIX))

        for blob in blobs:
            content = blob.download_as_bytes()
            img = Image.open(io.BytesIO(content))
            score = self.calculate_entropy(img)

            if score >= self.ENTROPY_THRESHOLD:
                out = f"{self.CLEANED_IMAGES_PREFIX}/{os.path.basename(blob.name)}"
                bucket.blob(out).upload_from_file(io.BytesIO(content), content_type="image/jpeg")

    # ========================== RUN ==========================
    def run(self):
        self.verify_bucket()

        pdfs = self.list_pdfs(self.PDFS_INPUT_PREFIX)
        for pdf in pdfs:
            self.process_pdf(pdf, self.OUTPUT_IMAGES_PREFIX)

        self.run_yolo_inference()
        self.run_object_cropping()
        self.run_image_cleaning()

        self.log("🏁 Pipeline completed successfully")
