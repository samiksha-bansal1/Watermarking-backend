import io
import json
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import os
import sys 

from embed import embed_image_watermark
from extract import DLWatermarkDecoder, extract_singular_values_from_image, extract_watermark_from_image_data

app = FastAPI(
    title="Image Watermark Embedding & Extraction API",
    description="API to embed and extract watermarks into images using DWT-SVD and Deep Learning",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Watermark Embedding Endpoint (Your existing code) ---
@app.post("/embed")
async def embed_watermark_api(
    image: UploadFile = File(...),
    watermark_length: int = Form(...),
    watermark_bits: str = Form(...),
    wavelet_type: str = Form("haar")
):
    try:
        # Parse watermark bits
        watermark_bits_list = json.loads(watermark_bits)
        if not isinstance(watermark_bits_list, list) or not all(b in [0, 1] for b in watermark_bits_list):
            raise HTTPException(status_code=400, detail="Watermark bits must be JSON array of 0s and 1s")

        print("\n=== EMBED REQUEST RECEIVED ===")
        print(f"Image filename           : {image.filename}")
        print(f"Watermark bit length     : {watermark_length}")
        print(f"Wavelet type             : {wavelet_type}")
        print(f"Watermark bits           : {watermark_bits_list}")
        print(f"Watermark as string      : {''.join(map(str, watermark_bits_list))}")

        # Read image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Embed watermark
        watermarked_rgb = embed_image_watermark(
            original_img_rgb_np=img_rgb,
            watermark_bit_length=watermark_length,
            watermark_bits_list=watermark_bits_list,
            wavelet_type=wavelet_type
        )

        if watermarked_rgb is None:
            raise HTTPException(status_code=500, detail="Watermark embedding failed")

        print("✅ Watermark embedded successfully.")

        # Encode result
        is_success, buffer = cv2.imencode(".png", cv2.cvtColor(watermarked_rgb, cv2.COLOR_RGB2BGR))
        if not is_success:
            raise HTTPException(status_code=500, detail="Failed to encode watermarked image")

        print("📦 Image encoded successfully and ready to return.\n")

        return StreamingResponse(
            io.BytesIO(buffer.tobytes()),
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=watermarked_image.png"}
        )

    except Exception as e:
        print(f"❌ ERROR during embedding: {str(e)}\n")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


# --- Watermark Extraction Configuration and Endpoint ---

# Use a base directory for cleaner path management
MODEL_DIR = "models" 
MODEL_NAME = "watermark_cnn_decoder_wm128_alpha0015.h5"
SCALER_NAME = "scaler_wm128_alpha0015.pkl" 

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
SCALER_PATH = os.path.join(MODEL_DIR, SCALER_NAME)

api_decoder = DLWatermarkDecoder(num_bits=128, wavelet_name='haar')

class WatermarkExtractionResponse(BaseModel):
    decoded_watermark: List[int]


# Startup event to load the model and scaler
@app.on_event("startup")
async def startup_event():
    """Load the DL watermark decoder model and scaler when the FastAPI application starts."""
    print("\n--- APP STARTUP: Loading DL watermark decoder model and scaler ---")
    try:
        full_model_path = os.path.abspath(MODEL_PATH)
        full_scaler_path = os.path.abspath(SCALER_PATH)

        if not os.path.exists(full_model_path):
            raise FileNotFoundError(f"Model file not found at: {full_model_path}")
        if not os.path.exists(full_scaler_path):
            raise FileNotFoundError(f"Scaler file not found at: {full_scaler_path}")

        # Call the load_components method from the DLWatermarkDecoder in extract.py
        api_decoder.load_components(MODEL_PATH, SCALER_PATH)
        print(f"DLWatermarkDecoder inferred input_seq_len: {api_decoder.input_seq_len}")
        print("✅ Model and scaler loaded successfully during startup.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR during startup: Failed to load model or scaler: {e}")
        sys.exit(1) # Exit the application if critical components can't be loaded



@app.post("/extract", response_model=WatermarkExtractionResponse)
async def extract_watermark_api(
    image: UploadFile = File(...),
    wavelet_type: str = Form("haar") # Allow overriding wavelet type for extraction if needed
):
    """
    Extracts a watermark from an uploaded image using the pre-trained DL model.
    """
    print("\n=== EXTRACT REQUEST RECEIVED ===")
    print(f"Image filename : {image.filename}")
    print(f"Wavelet type   : {wavelet_type}")

    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        # Read image content
        contents = await image.read()
        
        # Set the wavelet type for the decoder instance before decoding if it's different
        api_decoder.wavelet_name = wavelet_type

        # Decode the watermark using the function from extract.py
        decoded_watermark_bits = extract_watermark_from_image_data(contents, api_decoder)

        if decoded_watermark_bits is None:
            raise HTTPException(status_code=400, detail="Watermark extraction failed due to image processing error.")

        print(f"✅ Watermark extracted successfully. Decoded bits: {decoded_watermark_bits}")
        return {"decoded_watermark": decoded_watermark_bits}

    except HTTPException as e:
        print(f"❌ ERROR during extraction: {str(e)}\n")
        raise e # Re-raise FastAPI's HTTPExceptions directly
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR during extraction: {str(e)}\n")
        raise HTTPException(status_code=500, detail=f"Internal Server Error during extraction: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Watermark API is running"}