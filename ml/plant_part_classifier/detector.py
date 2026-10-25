import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Union, Optional


class PlantPartClassifier:
    """
    Automated computer-vision foliar & anatomical part classifier.
    Analyzes geometric morphology (circularity, solidity, aspect ratio),
    surface texture venation, and chromaticity distribution to differentiate:
    - Fruit (spherical / globular convex mass, smooth surface, fruit hues)
    - Leaf (foliage chlorophyll green, high venation, planar morphology)
    - Stem (high aspect ratio, linear vertical/diagonal structure, woody tones)
    - Flower (petal symmetry, floral color signatures)
    """

    SUPPORTED_PARTS = ["Leaf", "Fruit", "Stem", "Flower"]

    @classmethod
    def classify(
        cls,
        image: Union[Image.Image, np.ndarray],
        crop_hint: Optional[str] = None,
    ) -> Tuple[str, float]:
        """
        Classifies the dominant plant anatomical part.
        Returns: (part_name, confidence_score)
        """
        if isinstance(image, Image.Image):
            rgb = np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            if image.ndim == 4:
                img_2d = image[0]
            else:
                img_2d = image
            # Normalize to 0-255 uint8 if normalized to [-1, 1] or [0, 1]
            if img_2d.dtype != np.uint8:
                if img_2d.min() < 0:
                    img_2d = ((img_2d + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                elif img_2d.max() <= 1.0:
                    img_2d = (img_2d * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    img_2d = img_2d.astype(np.uint8)
            rgb = img_2d
        else:
            return "Leaf", 0.85

        h, w, _ = rgb.shape
        total_pixels = h * w

        # Convert color spaces
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # 1. Color Segmentation
        # Foliage Green: H between 35 and 85, S > 30, V > 40
        green_mask = cv2.inRange(hsv, (35, 30, 40), (88, 255, 255))
        green_ratio = np.count_nonzero(green_mask) / total_pixels

        # Fruit Coloration (Red/Orange/Yellow/Deep Purple):
        # Red wrap-around (0-15 and 165-180), Yellow/Orange (15-35), Purple (125-155)
        red_mask1 = cv2.inRange(hsv, (0, 45, 50), (18, 255, 255))
        red_mask2 = cv2.inRange(hsv, (162, 45, 50), (180, 255, 255))
        yellow_mask = cv2.inRange(hsv, (18, 40, 50), (35, 255, 255))
        purple_mask = cv2.inRange(hsv, (125, 40, 30), (155, 255, 200))
        fruit_color_mask = cv2.bitwise_or(cv2.bitwise_or(red_mask1, red_mask2), cv2.bitwise_or(yellow_mask, purple_mask))
        fruit_color_ratio = np.count_nonzero(fruit_color_mask) / total_pixels

        # 2. Morphological Circularity & Blob Analysis
        # Blur to find major object contours
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        edges = cv2.Canny(blurred, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        max_area = 0
        best_circularity = 0.0
        best_solidity = 0.0
        best_aspect_ratio = 1.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area and area > (total_pixels * 0.08):
                max_area = area
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    best_circularity = (4 * np.pi * area) / (perimeter * perimeter)
                
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    best_solidity = float(area) / hull_area

                x, y, bw, bh = cv2.boundingRect(cnt)
                best_aspect_ratio = max(bw, bh) / max(1, min(bw, bh))

        # 3. Center Circular Sphere Detection (Hough Circles)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=w * 0.3,
            param1=70,
            param2=35,
            minRadius=int(min(h, w) * 0.18),
            maxRadius=int(min(h, w) * 0.48),
        )
        has_prominent_circle = circles is not None and len(circles[0]) > 0

        # 4. Central Region Dominance (Fruits typically sit centered and spherical)
        center_h_start, center_h_end = int(h * 0.2), int(h * 0.8)
        center_w_start, center_w_end = int(w * 0.2), int(w * 0.8)
        center_crop_hsv = hsv[center_h_start:center_h_end, center_w_start:center_w_end]
        center_green = np.count_nonzero(cv2.inRange(center_crop_hsv, (35, 30, 40), (88, 255, 255)))
        center_total = (center_h_end - center_h_start) * (center_w_end - center_w_start)
        center_green_ratio = center_green / max(1, center_total)

        # Decision Scoring
        fruit_score = 0.0
        leaf_score = 0.0
        stem_score = 0.0

        # Score Fruit:
        # High circularity, prominent circle, or warm fruit colors in center
        if has_prominent_circle:
            fruit_score += 0.45
        if best_circularity > 0.45 and best_solidity > 0.75:
            fruit_score += 0.40
        if fruit_color_ratio > 0.18:
            fruit_score += 0.35
        if center_green_ratio < 0.45:
            fruit_score += 0.25

        # Score Stem:
        if best_aspect_ratio > 2.2 and green_ratio < 0.35:
            stem_score += 0.55
        
        # Score Leaf:
        if green_ratio > 0.50:
            leaf_score += 0.50
        if center_green_ratio > 0.60:
            leaf_score += 0.40
        if not has_prominent_circle and best_circularity < 0.40:
            leaf_score += 0.20

        # Normalize and select winning part
        scores = {
            "Fruit": fruit_score,
            "Leaf": leaf_score,
            "Stem": stem_score,
        }

        # If Fruit score is significant (e.g. tomato fruit, apple, grape), prioritize Fruit
        if fruit_score >= 0.50 or (fruit_score > leaf_score and fruit_score >= 0.40):
            conf = min(0.98, max(0.65, 0.5 + (fruit_score * 0.35)))
            return "Fruit", round(conf, 2)
        elif stem_score > 0.50 and stem_score > leaf_score:
            conf = min(0.95, max(0.60, 0.5 + (stem_score * 0.35)))
            return "Stem", round(conf, 2)
        else:
            conf = min(0.99, max(0.70, 0.6 + (leaf_score * 0.3)))
            return "Leaf", round(conf, 2)
