import numpy as np
import cv2
import pywt
import hashlib
import uuid


def generate_watermark(num_bits=128):
    """
    Generates a unique binary watermark by combining SHA256 hash of a UUID.
    """
    unique_id = str(uuid.uuid4())
    sha256_hash = hashlib.sha256(unique_id.encode('utf-8')).hexdigest()

    binary_hash = bin(int(sha256_hash, 16))[2:]

    if len(binary_hash) < num_bits:
        watermark_binary = binary_hash.zfill(num_bits)
    else:
        watermark_binary = binary_hash[:num_bits]

    return [int(bit) for bit in watermark_binary]


def apply_svd_embedding_and_reconstruct(subband_matrix, watermark_bits, alpha=0.005):
    U, S, Vt = np.linalg.svd(subband_matrix, full_matrices=False)
    S_prime = S.copy()

    num_embeddable_bits = min(len(S), len(watermark_bits))

    for i in range(num_embeddable_bits):
        if watermark_bits[i] == 1:
            S_prime[i] += alpha * S_prime[i]
        else:
            S_prime[i] -= alpha * S_prime[i]

    S_prime[S_prime < 0] = 0 

    modified_subband_matrix = U @ np.diag(S_prime) @ Vt
    return modified_subband_matrix, S, S_prime, U, Vt


def embed_watermark_core(original_img_rgb, watermark_bits, wavelet_name, alpha=0.01):
    R_orig = original_img_rgb[:, :, 0].astype(np.float32)
    G_orig = original_img_rgb[:, :, 1].astype(np.float32)
    B_orig = original_img_rgb[:, :, 2].astype(np.float32)

    coeffs_R_orig = pywt.dwt2(R_orig, wavelet_name)
    coeffs_G_orig = pywt.dwt2(G_orig, wavelet_name)
    coeffs_B_orig = pywt.dwt2(B_orig, wavelet_name)

    LL_R, (LH_R, HL_R, HH_R) = coeffs_R_orig
    LL_G, (LH_G, HL_G, HH_G) = coeffs_G_orig
    LL_B, (LH_B, HL_B, HH_B) = coeffs_B_orig

    subband_R_to_modify = LL_R.copy()
    subband_G_to_modify = LL_G.copy()
    subband_B_to_modify = LL_B.copy()

    subband_combined_orig = ((subband_R_to_modify + subband_G_to_modify + subband_B_to_modify) / 3.0).astype(np.float64)

    modified_subband_combined, S_original, S_watermarked, U_combined, Vt_combined = \
        apply_svd_embedding_and_reconstruct(subband_combined_orig, watermark_bits, alpha=alpha)

    subband_combined_orig_safe = np.where(subband_combined_orig == 0, 1e-9, subband_combined_orig)
    scaling_factor_map = modified_subband_combined / subband_combined_orig_safe

    watermarked_subband_R = subband_R_to_modify * scaling_factor_map
    watermarked_subband_G = subband_G_to_modify * scaling_factor_map
    watermarked_subband_B = subband_B_to_modify * scaling_factor_map

    coeffs_R_watermarked_final = list(coeffs_R_orig)
    coeffs_G_watermarked_final = list(coeffs_G_orig)
    coeffs_B_watermarked_final = list(coeffs_B_orig)

    coeffs_R_watermarked_final[0] = watermarked_subband_R
    coeffs_G_watermarked_final[0] = watermarked_subband_G
    coeffs_B_watermarked_final[0] = watermarked_subband_B

    coeffs_R_watermarked_final = (coeffs_R_watermarked_final[0], coeffs_R_watermarked_final[1])
    coeffs_G_watermarked_final = (coeffs_G_watermarked_final[0], coeffs_G_watermarked_final[1])
    coeffs_B_watermarked_final = (coeffs_B_watermarked_final[0], coeffs_B_watermarked_final[1])

    watermarked_R = pywt.idwt2(coeffs_R_watermarked_final, wavelet_name)
    watermarked_G = pywt.idwt2(coeffs_G_watermarked_final, wavelet_name)
    watermarked_B = pywt.idwt2(coeffs_B_watermarked_final, wavelet_name)

    h_orig, w_orig = original_img_rgb.shape[:2]
    watermarked_R = watermarked_R[:h_orig, :w_orig]
    watermarked_G = watermarked_G[:h_orig, :w_orig]
    watermarked_B = watermarked_B[:h_orig, :w_orig]

    watermarked_img_rgb = np.stack([watermarked_R, watermarked_G, watermarked_B], axis=-1)
    watermarked_img_rgb = np.clip(watermarked_img_rgb, 0, 255).astype(np.uint8)

    return watermarked_img_rgb, S_original, S_watermarked


def embed_image_watermark(original_img_rgb_np: np.ndarray, watermark_bit_length: int, watermark_bits_list: list, wavelet_type: str = 'haar') -> np.ndarray:
    if watermark_bit_length >= 64:
        embedding_alpha = 0.01
    else:
        embedding_alpha = 0.005

    if not watermark_bits_list:
        print("Warning: Watermark bits list is empty. No watermark will be embedded.")
        return original_img_rgb_np.copy()

    if len(watermark_bits_list) != watermark_bit_length:
        print(f"Warning: Actual watermark bits length ({len(watermark_bits_list)}) "
              f"differs from declared watermark_bit_length ({watermark_bit_length}).")

    try:
        watermarked_image_rgb, _, _ = embed_watermark_core(
            original_img_rgb_np, watermark_bits_list, wavelet_type, embedding_alpha
        )
        return watermarked_image_rgb
    except Exception as e:
        print(f"Error during watermark embedding: {e}")
        return None

