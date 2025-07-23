# import numpy as np
# import cv2
# import pywt 
# import tensorflow as tf
# import pickle
# from fastapi import HTTPException 
# from typing import List

# # --- Helper Functions ---

# def extract_singular_values_from_image(image_rgb: np.ndarray, wavelet_name: str):
#     """
#     Performs DWT and SVD on the LL subband of an image and returns
#     the singular values (S) from the combined R, G, B channels' LL subband.
#     """
#     R = image_rgb[:, :, 0].astype(np.float32)
#     G = image_rgb[:, :, 1].astype(np.float32)
#     B = image_rgb[:, :, 2].astype(np.float32)

#     coeffs_R = pywt.dwt2(R, wavelet_name)
#     coeffs_G = pywt.dwt2(G, wavelet_name)
#     coeffs_B = pywt.dwt2(B, wavelet_name)

#     LL_R, _ = coeffs_R
#     LL_G, _ = coeffs_G
#     LL_B, _ = coeffs_B

#     subband_combined = ((LL_R + LL_G + LL_B) / 3.0).astype(np.float64)

#     # Perform SVD on the combined subband
#     _, S, _ = np.linalg.svd(subband_combined, full_matrices=False)
#     return S

# # --- DL-based Decoder Class ---
# class DLWatermarkDecoder:
#     def __init__(self, num_bits: int = 128, wavelet_name: str = 'haar', input_seq_len: int = None):
#         self.num_bits = num_bits
#         self.wavelet_name = wavelet_name
#         self.scaler = None
#         self.model = None
#         self.input_seq_len = input_seq_len

#     def load_model_and_scaler(self, model_path: str, scaler_path: str):
#         """Loads a pre-trained model and scaler."""
#         try:
#             self.model = tf.keras.models.load_model(model_path)
#             print(f"Model loaded from {model_path}")
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=f"Failed to load Keras model from {model_path}: {e}")

#         try:
#             with open(scaler_path, 'rb') as f:
#                 self.scaler = pickle.load(f)
#             print(f"Scaler loaded from {scaler_path}")
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=f"Failed to load scaler from {scaler_path}: {e}")

#         # After loading, infer input_seq_len from the scaler or model if not explicitly set
#         if self.input_seq_len is None and hasattr(self.scaler, 'n_features_in_'):
#             self.input_seq_len = self.scaler.n_features_in_
#             print(f"Inferred input_seq_len from scaler: {self.input_seq_len}")
#         elif self.input_seq_len is None and self.model is not None:
#              # Attempt to get input shape from the first layer of the model
#             try:
#                 self.input_seq_len = self.model.layers[0].input_shape[1]
#                 print(f"Inferred input_seq_len from model: {self.input_seq_len}")
#             except Exception:
#                 pass # Will be caught by the final check if still None

#         if self.input_seq_len is None:
#             raise ValueError("Could not determine input_seq_len. It must be set either during initialization or derivable from loaded scaler/model.")


#     def decode_watermark(self, attacked_image_rgb: np.ndarray) -> List[int]:
#         """
#         Extracts and decodes the watermark from an attacked image using the trained DL model.
#         """
#         if self.model is None or self.scaler is None:
#             raise HTTPException(status_code=500, detail="Decoder model or scaler not loaded. Please load them first.")

#         if self.input_seq_len is None:
#             raise ValueError("Decoder input_seq_len is not set. Load the model and scaler first.")

#         S_extracted = extract_singular_values_from_image(attacked_image_rgb, self.wavelet_name)

#         current_num_bits = min(self.num_bits, len(S_extracted))

#         features_for_prediction = []
#         window_size = self.input_seq_len
#         window_half_size = window_size // 2

#         for i in range(current_num_bits):
#             start_idx = max(0, i - window_half_size)
#             end_idx = min(len(S_extracted), i + window_half_size + 1)

#             singular_value_window = S_extracted[start_idx:end_idx]

#             # PAD or TRUNCATE to ensure consistent fixed_input_seq_len for prediction
#             if len(singular_value_window) < window_size:
#                 padded_window = np.pad(singular_value_window, (0, window_size - len(singular_value_window)), 'constant')
#                 features_for_prediction.append(padded_window)
#             elif len(singular_value_window) > window_size:
#                 features_for_prediction.append(singular_value_window[:window_size])
#             else:
#                 features_for_prediction.append(singular_value_window)

#         features_for_prediction = np.array(features_for_prediction)

#         if features_for_prediction.shape[0] == 0:
#             return []

#         if features_for_prediction.shape[1] != self.input_seq_len:
#             print(f"Warning: Feature length mismatch for decoding. Expected {self.input_seq_len}, got {features_for_prediction.shape[1]}. Attempting to reshape.")
#             if features_for_prediction.shape[1] < self.input_seq_len:
#                 temp_features = np.zeros((features_for_prediction.shape[0], self.input_seq_len))
#                 temp_features[:, :features_for_prediction.shape[1]] = features_for_prediction
#                 features_for_prediction = temp_features
#             elif features_for_prediction.shape[1] > self.input_seq_len:
#                 features_for_prediction = features_for_prediction[:, :self.input_seq_len]


#         try:
#             features_scaled = self.scaler.transform(features_for_prediction.reshape(-1, self.input_seq_len))
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=f"Error during scaling features for decoding: {e}")

#         features_scaled = features_scaled.reshape(-1, self.input_seq_len, 1)

#         predictions = self.model.predict(features_scaled, verbose=0)
#         decoded_watermark = (predictions > 0.5).astype(int).flatten().tolist()

#         return decoded_watermark




import numpy as np
import cv2
import pywt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle
import os

# --- Helper Functions ---

def extract_singular_values_from_image(image_rgb, wavelet_name):
    """
    Performs DWT and SVD on the LL subband of an image and returns
    the singular values (S) from the combined R, G, B channels' LL subband.
    """
    R = image_rgb[:, :, 0].astype(np.float32)
    G = image_rgb[:, :, 1].astype(np.float32)
    B = image_rgb[:, :, 2].astype(np.float32)

    coeffs_R = pywt.dwt2(R, wavelet_name)
    coeffs_G = pywt.dwt2(G, wavelet_name)
    coeffs_B = pywt.dwt2(B, wavelet_name)

    LL_R, _ = coeffs_R
    LL_G, _ = coeffs_G
    LL_B, _ = coeffs_B

    subband_combined = ((LL_R + LL_G + LL_B) / 3.0).astype(np.float64)

    # Perform SVD on the combined subband
    _, S, _ = np.linalg.svd(subband_combined, full_matrices=False)
    return S

# --- DL-based Decoder Class ---
class DLWatermarkDecoder:
    def __init__(self, num_bits=128, wavelet_name='haar', input_seq_len=None):
        self.num_bits = num_bits
        self.wavelet_name = wavelet_name
        self.scaler = StandardScaler()
        self.model = None
        self.input_seq_len = input_seq_len

    def load_components(self, model_path, scaler_path):
        """
        Loads a pre-trained model and sets scaler parameters from saved files.
        """
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            self.model = None
            return False

        try:
            with open(scaler_path, 'rb') as f:
                scaler_params = pickle.load(f)
            self.scaler.mean_ = scaler_params['mean']
            self.scaler.scale_ = scaler_params['scale']
            self.scaler.n_features_in_ = scaler_params['n_features_in']
            self.input_seq_len = scaler_params['n_features_in'] # Set input_seq_len from scaler
            print(f"Scaler parameters loaded successfully from {scaler_path}")
        except Exception as e:
            print(f"Error loading scaler from {scaler_path}: {e}")
            return False
        
        return True

    def decode_watermark(self, attacked_image_rgb):
        """
        Extracts and decodes the watermark from an attacked image using the trained DL model.
        """
        if self.model is None or self.input_seq_len is None:
            print("Error: Model or scaler not loaded. Please call load_components first.")
            return []

        S_extracted = extract_singular_values_from_image(attacked_image_rgb, self.wavelet_name)

        current_num_bits = min(self.num_bits, len(S_extracted))

        features_for_prediction = []
        window_size = self.input_seq_len # Use the loaded input_seq_len
        window_half_size = window_size // 2

        for i in range(current_num_bits):
            start_idx = max(0, i - window_half_size)
            end_idx = min(len(S_extracted), i + window_half_size + 1)

            singular_value_window = S_extracted[start_idx:end_idx]

            # PAD or TRUNCATE to ensure consistent fixed_input_seq_len for prediction
            if len(singular_value_window) < window_size:
                padded_window = np.pad(singular_value_window, (0, window_size - len(singular_value_window)), 'constant')
                features_for_prediction.append(padded_window)
            elif len(singular_value_window) > window_size:
                features_for_prediction.append(singular_value_window[:window_size])
            else:
                features_for_prediction.append(singular_value_window)

        features_for_prediction = np.array(features_for_prediction)

        if features_for_prediction.shape[0] == 0:
            return []

        # Ensure the feature length matches the expected input_seq_len
        if features_for_prediction.shape[1] != self.input_seq_len:
            print(f"Warning: Feature length mismatch for decoding. Expected {self.input_seq_len}, got {features_for_prediction.shape[1]}. Attempting to reshape.")
            if features_for_prediction.shape[1] < self.input_seq_len:
                temp_features = np.zeros((features_for_prediction.shape[0], self.input_seq_len))
                temp_features[:, :features_for_prediction.shape[1]] = features_for_prediction
                features_for_prediction = temp_features
            elif features_for_prediction.shape[1] > self.input_seq_len:
                features_for_prediction = features_for_prediction[:, :self.input_seq_len]

        try:
            features_scaled = self.scaler.transform(features_for_prediction.reshape(-1, self.input_seq_len))
        except Exception as e:
            print(f"Error during scaling: {e}. This might indicate scaler was not fitted correctly or input shape mismatch. "
                  f"Features shape: {features_for_prediction.shape}, Expected scaler input shape: (-1, {self.input_seq_len})")
            return []

        features_scaled = features_scaled.reshape(-1, self.input_seq_len, 1)

        predictions = self.model.predict(features_scaled, verbose=0)

        decoded_watermark = (predictions > 0.5).astype(int).flatten().tolist()

        return decoded_watermark

def extract_watermark_from_image_data(image_data, decoder):
    """
    Loads an image from byte data, preprocesses it, and extracts the watermark.
    """
    # Convert bytes to numpy array
    np_array = np.frombuffer(image_data, np.uint8)
    original_img_bgr = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if original_img_bgr is None:
        print("Error: Could not decode image data.")
        return None

    # Resize image to 512x512 if larger, consistent with training
    if original_img_bgr.shape[0] > 512 or original_img_bgr.shape[1] > 512:
        max_dim = 512
        scale = max_dim / max(original_img_bgr.shape[0], original_img_bgr.shape[1])
        original_img_bgr = cv2.resize(original_img_bgr, (int(original_img_bgr.shape[1] * scale), int(original_img_bgr.shape[0] * scale)))
    
    image_rgb = cv2.cvtColor(original_img_bgr, cv2.COLOR_BGR2RGB)
    
    print("Processing image data for extraction...")
    extracted_watermark = decoder.decode_watermark(image_rgb)
    return extracted_watermark

