"""
PlantScan AI - Foliar Lesion Detection and Highlighting Engine

Detects and visually highlights necrotic, chlorotic, and disease-damaged foliar lesions.
Strictly distinguishes Leaf Boundary Segmentation (leaf vs background) from
Lesion Segmentation (damaged tissue within the leaf boundary).

Calculates exact image-based agronomic metrics:
    - Total valid leaf pixels
    - Total lesion pixels inside leaf mask
    - Affected area percentage = (lesion_pixels / leaf_pixels) * 100
    - Number of distinct lesion regions
    - Bounding boxes around detected lesions
"""

import base64
import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image


@dataclass
class LesionMeasurement:
    lesion_detected: bool
    lesion_count: int
    affected_area_percentage: float
    leaf_area_pixels: int
    lesion_area_pixels: int
    severity_level: str
    bounding_boxes: List[Dict[str, int]] = field(default_factory=list)
    overlay_image_base64: Optional[str] = None
    mask_image_base64: Optional[str] = None
    method_note: str = (
        "Experimental Computer Vision Morphological & Local Contrast Analysis. "
        "This localized overlay is an experimental visualization aid highlighting candidate necrotic/chlorotic "
        "regions based on edge and chromaticity gradients; it is not a certified pixel-level ground-truth segmentation model."
    )


class PlantLesionDetector:
    """
    Two-stage computer vision lesion detector:
    Stage 1: Plant Tissue Mask Segmentation (separating leaf or fruit tissue from background).
    Stage 2: Lesion Segmentation within Tissue Mask (isolating localized necrotic & chlorotic lesions
             while strictly preserving normal healthy leaf and fruit coloration).
    """

    @staticmethod
    def segment_leaf(rgb_img: np.ndarray, is_fruit: bool = False) -> np.ndarray:
        """
        Segment the plant tissue (leaf or fruit) from image background.
        Returns binary mask (uint8, 255=tissue, 0=background).
        """
        h, w, _ = rgb_img.shape
        total_pixels = h * w

        hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        s_chan = hsv[:, :, 1]
        v_chan = hsv[:, :, 2]

        if not is_fruit:
            # Leaf mode: foliage is distinguished by saturation and green/yellowish chromaticity
            # Otsu threshold on saturation channel isolates foliage from neutral gray/white background sheets
            _, mask_otsu_s = cv2.threshold(s_chan, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask_green = cv2.inRange(hsv, (18, 25, 25), (95, 255, 255))
            mask_foliage = (mask_green | ((s_chan > 35) & (l_chan < 225) & (l_chan > 30)))
            combined = mask_foliage & (mask_otsu_s > 0)
        else:
            # Fruit foreground: colored tissue (red, orange, yellow, green) vs neutral sheet/glare
            is_fg = (l_chan < 245) & ((s_chan > 20) | (l_chan > 40)) & (v_chan > 20)
            combined = is_fg & ((s_chan > 25) | (a_chan > 135) | (b_chan > 135))

        raw_mask = combined.astype(np.uint8) * 255

        # Morphological filtering: close holes and remove background speckle
        kernel_lg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cleaned = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel_lg, iterations=2)
        kernel_sm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_sm, iterations=1)

        # Fill holes in the largest components
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tissue_mask = np.zeros_like(cleaned)
        min_tissue_area = total_pixels * 0.02

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_tissue_area:
                cv2.drawContours(tissue_mask, [cnt], -1, 255, thickness=cv2.FILLED)

        # If no single contour passed threshold, fall back to cleaned mask if sufficiently large
        if np.count_nonzero(tissue_mask) == 0 and np.count_nonzero(cleaned) >= min_tissue_area:
            tissue_mask = cleaned

        return tissue_mask

    segment_plant_tissue = segment_leaf

    @classmethod
    def segment_lesions(
        cls,
        rgb_img: np.ndarray,
        tissue_mask: np.ndarray,
        is_fruit: bool = False,
        is_healthy: bool = False,
    ) -> np.ndarray:
        """
        Segment lesion and necrotic patches strictly within the segmented plant tissue mask.
        Prevents normal fruit color (red, yellow, orange) and normal green foliage from being marked as lesions.
        Returns binary mask (uint8, 255=lesion, 0=healthy/background).
        """
        total_tissue_pixels = np.count_nonzero(tissue_mask)
        if total_tissue_pixels < 100:
            return np.zeros_like(tissue_mask)

        hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        h_chan, s_chan, v_chan = cv2.split(hsv)

        # Erode boundary slightly (5x5) to eliminate boundary shadow edge artifacts
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        inner_mask = cv2.erode(tissue_mask, kernel_erode)

        # Compute local luminance and local gradient
        # True lesions represent sharp local drops and edge discontinuities, not smooth gradual shading
        blur_l = cv2.GaussianBlur(l_chan, (25, 25), 0)
        local_drop = blur_l.astype(np.float32) - l_chan.astype(np.float32)

        laplacian = cv2.Laplacian(l_chan, cv2.CV_32F, ksize=3)
        edge_mag = np.abs(laplacian)

        # Reject specular reflections / glare hotspots (high L, low saturation)
        specular_glare = (l_chan > 220) & (s_chan < 55)

        if is_fruit:
            # Fruit tissue:
            # Normal fruit skin (smooth red, yellow, orange, or green) is healthy and uniform.
            # Only localized dark sunken necrotic rot, fungal decay, or black pits are lesions.
            is_vivid_red = (a_chan > 142) & (s_chan > 60)
            is_vivid_yellow = (l_chan > 100) & (b_chan > 132)
            is_vivid_green = (a_chan < 124) & (l_chan > 80)
            is_healthy_skin = is_vivid_red | is_vivid_yellow | is_vivid_green

            fruit_lesions = (
                (inner_mask == 255) &
                (~specular_glare) &
                (~is_healthy_skin) &
                (
                    (l_chan < 45) |
                    ((local_drop > 22.0) & (l_chan < 90)) |
                    ((l_chan < 75) & (s_chan < 70))
                )
            )
            raw_lesions = fruit_lesions.astype(np.uint8) * 255
        else:
            # Leaf tissue:
            # Healthy photosynthetic green foliage must be strictly protected from necrotic false positives
            is_healthy_green = (h_chan >= 32) & (h_chan <= 88) & (a_chan < 124)

            # 1. Necrotic spots (brown/black/grey tissue decay):
            necrotic = (
                (inner_mask == 255) &
                (~specular_glare) &
                (~is_healthy_green) &
                (
                    # Dark necrotic spot with local contrast drop
                    ((local_drop > 16.0) & (l_chan < 105)) |
                    # Severe dark necrosis
                    ((l_chan < 65) & (local_drop > 10.0)) |
                    # Distinct brown/reddish blight lesion with sharp boundary edge
                    ((a_chan > 132) & (local_drop > 12.0) & (edge_mag > 5.0))
                )
            )

            # 2. Chlorotic lesions (localized yellowing/rust spots with sharp edge contrast against green)
            chlorotic = (
                (inner_mask == 255) &
                (~specular_glare) &
                (~is_healthy_green) &
                (h_chan >= 12) & (h_chan <= 30) & (s_chan > 75) &
                (local_drop > 12.0) & (edge_mag > 6.0)
            )

            raw_lesions = (necrotic | chlorotic).astype(np.uint8) * 255

        # Morphological noise filtering
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        filtered = cv2.morphologyEx(raw_lesions, cv2.MORPH_OPEN, kernel_open)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        connected = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel_close)

        # Filter lesion components: must be within tissue and have meaningful spot size
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_lesion_mask = np.zeros_like(tissue_mask)

        max_valid_lesion_area = total_tissue_pixels * 0.65
        min_valid_lesion_area = 24 if is_healthy else 12  # Higher threshold for verified healthy specimens

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_valid_lesion_area <= area <= max_valid_lesion_area:
                cv2.drawContours(final_lesion_mask, [cnt], -1, 255, thickness=cv2.FILLED)

        # If specimen is predicted healthy and total detected lesion area is negligible (< 1.5%),
        # suppress isolated edge/fold noise.
        if is_healthy:
            lesion_px = np.count_nonzero(final_lesion_mask)
            if (lesion_px / max(1, total_tissue_pixels)) < 0.015:
                final_lesion_mask = np.zeros_like(tissue_mask)

        return final_lesion_mask

    @classmethod
    def analyze(
        cls,
        image_input: Union[Image.Image, np.ndarray],
        plant_part: str = "Leaf",
        is_healthy: Optional[bool] = None,
    ) -> LesionMeasurement:
        """
        Runs full lesion detection pipeline on input image.
        Returns LesionMeasurement containing metrics, overlay base64, and mask base64.
        """
        if isinstance(image_input, Image.Image):
            rgb_img = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 4:
                rgb_img = image_input[0]
            else:
                rgb_img = image_input
            if rgb_img.dtype != np.uint8:
                if rgb_img.max() <= 1.0:
                    rgb_img = (rgb_img * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    rgb_img = rgb_img.clip(0, 255).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        is_fruit = str(plant_part).strip().lower() == "fruit"

        # 1. Plant Tissue Segmentation (Leaf or Fruit)
        tissue_mask = cls.segment_leaf(rgb_img, is_fruit=is_fruit)
        tissue_pixels = int(np.count_nonzero(tissue_mask))
        total_img_pixels = rgb_img.shape[0] * rgb_img.shape[1]

        # Check if valid plant tissue is detected
        if tissue_pixels < (total_img_pixels * 0.02):
            part_name = "Fruit" if is_fruit else "Leaf"
            return LesionMeasurement(
                lesion_detected=False,
                lesion_count=0,
                affected_area_percentage=0.0,
                leaf_area_pixels=0,
                lesion_area_pixels=0,
                severity_level=f"None (No {part_name} Detected)",
                method_note=f"No prominent plant {part_name.lower()} tissue detected in the uploaded image.",
            )

        # 2. Lesion Segmentation
        lesion_mask = cls.segment_lesions(
            rgb_img,
            tissue_mask,
            is_fruit=is_fruit,
            is_healthy=bool(is_healthy),
        )
        lesion_pixels = int(np.count_nonzero(lesion_mask))

        # Calculate Affected Area Percentage
        affected_pct = round((lesion_pixels / max(1, tissue_pixels)) * 100.0, 2)

        # Contours and Bounding Boxes
        contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = int(cv2.contourArea(cnt))
            if area >= 12:
                bounding_boxes.append({"x": int(x), "y": int(y), "width": int(w), "height": int(h), "area": area})

        lesion_count = len(bounding_boxes)
        lesion_detected = lesion_count > 0 and lesion_pixels >= 10

        # Severity level classification
        if not lesion_detected or affected_pct < 0.2:
            severity = "None (Healthy)"
        elif affected_pct < 5.0:
            severity = "Low (<5%)"
        elif affected_pct <= 20.0:
            severity = "Moderate (5-20%)"
        else:
            severity = "Severe (>20%)"

        # 3. Visual Overlay Generation
        overlay = rgb_img.copy()
        red_tint = np.zeros_like(rgb_img)
        red_tint[:] = [239, 68, 68]  # Agricultural warning red

        lesion_bool = lesion_mask == 255
        overlay[lesion_bool] = (
            rgb_img[lesion_bool] * 0.55 + red_tint[lesion_bool] * 0.45
        ).astype(np.uint8)

        # Draw red contour outlines around lesions
        cv2.drawContours(overlay, contours, -1, (220, 38, 38), thickness=2)

        # Convert Overlay to Base64 JPEG
        overlay_pil = Image.fromarray(overlay)
        buf_overlay = io.BytesIO()
        overlay_pil.save(buf_overlay, format="JPEG", quality=90)
        overlay_b64 = "data:image/jpeg;base64," + base64.b64encode(buf_overlay.getvalue()).decode("utf-8")

        # Convert Binary Mask to Base64 PNG
        mask_pil = Image.fromarray(lesion_mask)
        buf_mask = io.BytesIO()
        mask_pil.save(buf_mask, format="PNG")
        mask_b64 = "data:image/png;base64," + base64.b64encode(buf_mask.getvalue()).decode("utf-8")

        return LesionMeasurement(
            lesion_detected=lesion_detected,
            lesion_count=lesion_count,
            affected_area_percentage=affected_pct,
            leaf_area_pixels=tissue_pixels,
            lesion_area_pixels=lesion_pixels,
            severity_level=severity,
            bounding_boxes=bounding_boxes,
            overlay_image_base64=overlay_b64,
            mask_image_base64=mask_b64,
        )
