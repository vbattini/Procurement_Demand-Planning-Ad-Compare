# pdf_processing_pipeline.py
import os
import io
import time
import cv2
import json
import numpy as np
import csv
import uuid
from PIL import Image
from pathlib import Path
from pdf2image import convert_from_bytes
from datetime import datetime
from scipy.stats import entropy
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from google.cloud import storage
from ultralytics import YOLO

import logging
import os
if os.system("which pdfinfo > /dev/null 2>&1") != 0:
    print("🔧 Installing poppler-utils...")
    os.system("apt-get install -y poppler-utils")

# Disable Ultralytics banner and limit log level
os.environ["ULTRALYTICS_VERBOSE"] = "False"
os.environ["ULTRALYTICS_HIDE_CONFIG_WARNINGS"] = "True"

# Suppress Python + Ultralytics logs
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("engine").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

class PDFProcessingPipeline:
    def __init__(self, gcs_bucket_name: str, input_pdf: str, pipeline_root_prefix: str = "DPAC/8.TestRun"):
        self.GCS_BUCKET_NAME = gcs_bucket_name
        self.INPUT_PDF = input_pdf
        self.GCS_PIPELINE_ROOT_PREFIX = pipeline_root_prefix

        # Derived paths
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

        # Timer
        self.PIPELINE_START = datetime.now()

    # ----------------------------
    # Utility
    # ----------------------------
    def log(self, msg: str):
        elapsed = datetime.now() - self.PIPELINE_START
        m, s = divmod(elapsed.total_seconds(), 60)
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} | ⏱ {int(m)}m {int(s)}s | {msg}")
    # def log(self, msg: str):
    #   if self.verbose:  # only log when verbose=True
    #     elapsed = datetime.now() - self.PIPELINE_START
    #     m, s = divmod(elapsed.total_seconds(), 60)
    #     print(f"{datetime.now():%Y-%m-%d %H:%M:%S} | ⏱ {int(m)}m {int(s)}s | {msg}")


    def init_gcs_client(self):
        return storage.Client()

    def verify_bucket(self):
        client = self.init_gcs_client()
        bucket = client.bucket(self.GCS_BUCKET_NAME)
        if not bucket.exists():
            raise ValueError(f"Bucket '{self.GCS_BUCKET_NAME}' does not exist.")
        _ = list(bucket.list_blobs(prefix="", max_results=1))
        self.log(f"✅ Bucket '{self.GCS_BUCKET_NAME}' exists and is accessible.")
        return bucket

    def download_blob(self, source_blob_name, dest_file):
        self.init_gcs_client().bucket(self.GCS_BUCKET_NAME).blob(source_blob_name).download_to_filename(dest_file)

    def upload_blob(self, local_file, dest_blob_name):
        self.init_gcs_client().bucket(self.GCS_BUCKET_NAME).blob(dest_blob_name).upload_from_filename(local_file)

    def list_pdfs(self, prefix):
        client = self.init_gcs_client()
        bucket = client.bucket(self.GCS_BUCKET_NAME)
        return [b.name for b in bucket.list_blobs(prefix=prefix) if b.name.lower().endswith(".pdf")]

    # ----------------------------
    # Step 1: PDF → Image Conversion
    # ----------------------------
    def convert_pdf_page_to_jpg(self, pdf_bytes, pdf_basename, output_prefix, page_number):
        bucket = self.init_gcs_client().bucket(self.GCS_BUCKET_NAME)
        try:
            images = convert_from_bytes(pdf_bytes, dpi=self.DPI, fmt="jpeg",
                                        first_page=page_number, last_page=page_number, thread_count=1)
            img = images[0]
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=self.JPEG_QUALITY, optimize=True)
            buf.seek(0)
            filename = f"{pdf_basename}_page_{page_number}.jpg"
            bucket.blob(f"{output_prefix}/{filename}").upload_from_file(buf, content_type="image/jpeg")
            return 1
        except Exception as e:
            self.log(f"❌ Failed page {page_number} of {pdf_basename}: {e}")
            return 0

    def process_pdf(self, pdf_blob_name, output_prefix):
        client = self.init_gcs_client()
        bucket = client.bucket(self.GCS_BUCKET_NAME)
        blob = bucket.blob(pdf_blob_name)
        pdf_bytes = blob.download_as_bytes()
        pdf_basename = os.path.splitext(os.path.basename(pdf_blob_name))[0]

        images = convert_from_bytes(pdf_bytes, dpi=72, fmt="jpeg", thread_count=1)
        total_pages = len(images)
        del images

        pages_converted = 0
        with ProcessPoolExecutor(max_workers=min(self.MAX_WORKERS, total_pages)) as executor:
            futures = {executor.submit(self.convert_pdf_page_to_jpg, pdf_bytes, pdf_basename, output_prefix, i + 1): i + 1 for i in range(total_pages)}
            for f in futures:
                pages_converted += f.result()
        return pages_converted

    # ----------------------------
    # Step 2: YOLO Inference
    # ----------------------------



    def run_yolo_inference(self):
        self.download_blob(self.TRAINED_MODEL_FILENAME, self.LOCAL_MODEL_PATH)
        model = YOLO(self.LOCAL_MODEL_PATH)
        bucket = self.init_gcs_client().bucket(self.GCS_BUCKET_NAME)

        total_images = 0
        total_detections = 0

        for blob in bucket.list_blobs(prefix=self.TEST_IMAGES_PREFIX):
            if blob.name.lower().endswith((".jpg", ".jpeg", ".png")):
                filename = os.path.basename(blob.name)
                local_input = f"/tmp/{filename}"
                local_output = f"/tmp/out_{filename}"
                blob.download_to_filename(local_input)

                color_img = cv2.imread(local_input)
                input_img = (
                    cv2.cvtColor(cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                    if self.USE_GRAYSCALE else color_img.copy()
                )

                results = model.predict(source=input_img, imgsz=self.IMG_SIZE, conf=0.4, verbose=False)
                num_boxes = len(results[0].boxes)
                total_images += 1
                total_detections += num_boxes
                print(f"🖼️ {filename}: {num_boxes} detections")

                annotated = results[0].plot(img=color_img.copy(), line_width=self.LINE_WIDTH)
                cv2.imwrite(local_output, annotated)
                self.upload_blob(local_output, f"{self.PREDICTION_RESULTS_PREFIX}/{filename}")

        print(f"\n✅ Processed {total_images} images")
        print(f"📦 Total detections across all images: {total_detections}")


    def download_all_images(self, prefix, local_dir):
        client = self.init_gcs_client()
        bucket = client.bucket(self.GCS_BUCKET_NAME)
        image_paths = []

        os.makedirs(local_dir, exist_ok=True)
        for blob in bucket.list_blobs(prefix=prefix):
            if blob.name.endswith("/") or not blob.name.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
            ):
                continue

            local_path = os.path.join(local_dir, os.path.basename(blob.name))
            blob.download_to_filename(local_path)
            image_paths.append(local_path)

        return image_paths


    def process_image_crop(self, image_path, model, skipped_log_writer):
        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️ Skipping unreadable image: {image_path}")
            return 0, []

        h, w = img.shape[:2]
        results = model.predict(img, conf=0.4, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().tolist()

        cropped_count = 0
        skipped_boxes = []

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(float, box)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1 or (x2 - x1) < 3 or (y2 - y1) < 3:
                reason = "Tiny or invalid box"
                skipped_boxes.append((idx + 1, box, reason))
                skipped_log_writer.writerow([
                    os.path.basename(image_path), idx + 1,
                    round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2),
                    reason
                ])
                continue

            cropped = img[int(y1):int(y2), int(x1):int(x2)]
            if cropped.size == 0:
                reason = "Empty crop"
                skipped_boxes.append((idx + 1, box, reason))
                skipped_log_writer.writerow([
                    os.path.basename(image_path), idx + 1,
                    round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2),
                    reason
                ])
                continue

            filename = os.path.splitext(os.path.basename(image_path))[0]
            crop_filename = f"{filename}_crop_{idx+1}.jpg"
            local_crop_path = os.path.join(self.LOCAL_CROP_DIR, crop_filename)
            os.makedirs(self.LOCAL_CROP_DIR, exist_ok=True)
            cv2.imwrite(local_crop_path, cropped)
            cropped_count += 1

            gcs_upload_path = f"{self.CROPPED_OBJECTS_PREFIX}/{crop_filename}"
            self.upload_blob(local_crop_path, gcs_upload_path)

        page_name = os.path.basename(image_path)
        print(f"🖼️ {page_name} → {cropped_count} cropped objects (skipped {len(skipped_boxes)})")
        return cropped_count, skipped_boxes


    def run_object_cropping(self):
        print("🚀 Starting YOLO object cropping...")
        self.download_blob(self.TRAINED_MODEL_FILENAME, self.LOCAL_MODEL_PATH)
        model = YOLO(self.LOCAL_MODEL_PATH)

        image_paths = self.download_all_images(self.TEST_IMAGES_PREFIX, self.LOCAL_IMAGE_DOWNLOAD_DIR)
        if not image_paths:
            print("⚠️ No images found for cropping.")
            return

        total_crops = 0
        total_skipped = 0
        skipped_log_path = os.path.join(self.LOCAL_CROP_DIR, "skipped_detections.csv")
        os.makedirs(self.LOCAL_CROP_DIR, exist_ok=True)

        with open(skipped_log_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Image_Name", "Detection_Index", "x1", "y1", "x2", "y2", "Reason"])

            for img_path in image_paths:
                cropped_count, skipped_boxes = self.process_image_crop(img_path, model, writer)
                total_crops += cropped_count
                total_skipped += len(skipped_boxes)

        print(f"\n📊 SUMMARY: {total_crops} cropped objects, {total_skipped} skipped")
        print(f"📄 Detailed skip log saved → {skipped_log_path}")
        # ----------------------------
        # Step 4: Noise Filtering
        # ----------------------------
    def calculate_entropy(self, image: Image.Image) -> float:
        img_gray = image.convert("L")
        hist, _ = np.histogram(np.array(img_gray).flatten(), bins=256, range=(0, 255), density=True)
        hist = hist[hist > 0]
        return entropy(hist)

    def process_single_image_noise(self, bucket, blob):
        if not blob.name.lower().endswith(('jpg', 'jpeg', 'png')):
            return None
        try:
            content = blob.download_as_bytes()
            img = Image.open(io.BytesIO(content))
            score = self.calculate_entropy(img)
            if score < self.ENTROPY_THRESHOLD:
                self.log(f"🗑️ Noisy image: {blob.name} | Entropy: {score:.2f}")
                return "noisy"
            else:
                output_path = f"{self.CLEANED_IMAGES_PREFIX}/{os.path.basename(blob.name)}"
                bucket.blob(output_path).upload_from_file(io.BytesIO(content), content_type="image/jpeg")
                return "clean"
        except Exception as e:
            self.log(f"❌ Error processing {blob.name}: {e}")
            return None

    def run_image_cleaning(self):
        client = self.init_gcs_client()
        bucket = client.bucket(self.GCS_BUCKET_NAME)
        blobs = list(bucket.list_blobs(prefix=self.CROPPED_OBJECTS_PREFIX))
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS_NOISE) as executor:
            list(executor.map(lambda b: self.process_single_image_noise(bucket, b), blobs))
    def clear_gcs_prefix(self, prefix: str, batch_size: int = 100, workers: int = 20):
      """
      Deletes BEFORE uploading new data.
      """
      client = self.init_gcs_client()
      bucket = client.bucket(self.GCS_BUCKET_NAME)
      blobs = list(bucket.list_blobs(prefix=prefix))
      if not blobs:
          self.log(f"ℹ️ No files to delete in {prefix}")
          return

      self.log(f"🧹 Deleting {len(blobs)} files from {prefix}")
      def delete_batch(batch):
          with client.batch():
              for blob in batch:
                  blob.delete()

      batches = [
          blobs[i:i + batch_size]
          for i in range(0, len(blobs), batch_size)
      ]
      with ThreadPoolExecutor(max_workers=workers) as executor:
          executor.map(delete_batch, batches)

      self.log(f"✅ Cleared folder: {prefix}")
    # ----------------------------
    # Pipeline Orchestration
    # ----------------------------
    def run(self):
        pipeline_start = time.time()
        # self.log("Step 1: Verify GCS bucket...")
        self.verify_bucket()
        #Delete immediately BEFORE writing to that folder:
        self.clear_gcs_prefix(self.OUTPUT_IMAGES_PREFIX)
        self.clear_gcs_prefix(self.PREDICTION_RESULTS_PREFIX)
        self.clear_gcs_prefix(self.CROPPED_OBJECTS_PREFIX)
        self.clear_gcs_prefix(self.CLEANED_IMAGES_PREFIX)
        pdf_files = self.list_pdfs(self.PDFS_INPUT_PREFIX)
        if pdf_files:
            self.log(f"Step 2: Found {len(pdf_files)} PDF(s) to convert.")
            for pdf_name in pdf_files:
                self.process_pdf(pdf_name, self.OUTPUT_IMAGES_PREFIX)
        # self.log("Step 3: Running YOLO inference...")
        self.run_yolo_inference()
        # self.log("Step 4: Running object cropping...")
        self.run_object_cropping()
        # self.log("Step 5: Running noise filtering / image cleaning...")
        self.run_image_cleaning()
        self.log(f"🏁 Pipeline complete | Total time: {time.time() - pipeline_start:.2f}s")
