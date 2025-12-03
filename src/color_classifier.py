"""
Color Classifier Module

Classifies extracted RGB colors into standard tattoo ink categories.
Uses color distance calculations in LAB color space for perceptual accuracy.
"""

import numpy as np
from typing import Tuple, Dict, List


class ColorClassifier:
    """
    Classifies RGB colors into tattoo ink categories based on
    perceptual color distance in LAB color space.
    """

    # Reference colors for tattoo ink categories (RGB)
    # Values represent typical ink pigments observed in professional tattoos
    INK_REFERENCES: Dict[str, Tuple[int, int, int]] = {
        "black": (20, 20, 20),
        "dark_gray": (60, 60, 60),
        "gray": (120, 120, 120),
        "blue": (30, 60, 150),
        "light_blue": (70, 130, 200),
        "turquoise": (0, 150, 150),
        "green": (30, 100, 50),
        "teal": (0, 120, 100),
        "red": (180, 40, 40),
        "orange": (220, 100, 40),
        "yellow": (230, 200, 50),
        "purple": (100, 40, 120),
        "pink": (220, 120, 150),
        "white": (240, 240, 240),
        "brown": (100, 60, 40),
    }

    # Primary category mappings for simplified output
    CATEGORY_MAP: Dict[str, str] = {
        "black": "black",
        "dark_gray": "black",
        "gray": "black",
        "blue": "blue",
        "light_blue": "blue",
        "turquoise": "turquoise",
        "green": "green",
        "teal": "green",
        "red": "red",
        "orange": "red",
        "yellow": "yellow",
        "purple": "purple",
        "pink": "red",
        "white": "white",
        "brown": "brown",
    }

    def __init__(self, skin_threshold: float = 0.3):
        """
        Initialize the color classifier.

        Args:
            skin_threshold: Threshold for filtering potential skin tones (0-1).
                           Higher values = more aggressive skin filtering.
        """
        self.skin_threshold = skin_threshold
        self._build_reference_lab()

    def _build_reference_lab(self) -> None:
        """Pre-compute LAB values for reference colors."""
        self._reference_lab: Dict[str, np.ndarray] = {}
        for name, rgb in self.INK_REFERENCES.items():
            self._reference_lab[name] = self._rgb_to_lab(np.array(rgb))

    @staticmethod
    def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
        """
        Convert RGB to LAB color space.

        Uses D65 illuminant for conversion. LAB provides perceptually
        uniform color distances.

        Args:
            rgb: RGB values as numpy array (0-255 range)

        Returns:
            LAB values as numpy array
        """
        # Normalize RGB to 0-1
        rgb_norm = rgb.astype(np.float64) / 255.0

        # Apply gamma correction (sRGB to linear)
        mask = rgb_norm > 0.04045
        rgb_linear = np.where(
            mask,
            np.power((rgb_norm + 0.055) / 1.055, 2.4),
            rgb_norm / 12.92
        )

        # Convert to XYZ (D65 illuminant)
        matrix = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        xyz = np.dot(matrix, rgb_linear)

        # Normalize by D65 white point
        xyz_ref = np.array([0.95047, 1.0, 1.08883])
        xyz_norm = xyz / xyz_ref

        # Convert to LAB
        epsilon = 0.008856
        kappa = 903.3

        f_xyz = np.where(
            xyz_norm > epsilon,
            np.power(xyz_norm, 1/3),
            (kappa * xyz_norm + 16) / 116
        )

        L = 116 * f_xyz[1] - 16
        a = 500 * (f_xyz[0] - f_xyz[1])
        b = 200 * (f_xyz[1] - f_xyz[2])

        return np.array([L, a, b])

    @staticmethod
    def _delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> float:
        """
        Calculate CIEDE2000 color difference.

        This is the most perceptually accurate color distance metric,
        accounting for human vision characteristics.

        Args:
            lab1: First LAB color
            lab2: Second LAB color

        Returns:
            Delta E 2000 value (0 = identical, >100 = very different)
        """
        L1, a1, b1 = lab1
        L2, a2, b2 = lab2

        # Calculate C and h
        C1 = np.sqrt(a1**2 + b1**2)
        C2 = np.sqrt(a2**2 + b2**2)
        C_avg = (C1 + C2) / 2

        G = 0.5 * (1 - np.sqrt(C_avg**7 / (C_avg**7 + 25**7)))

        a1_prime = a1 * (1 + G)
        a2_prime = a2 * (1 + G)

        C1_prime = np.sqrt(a1_prime**2 + b1**2)
        C2_prime = np.sqrt(a2_prime**2 + b2**2)

        h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360
        h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360

        # Calculate deltas
        delta_L = L2 - L1
        delta_C = C2_prime - C1_prime

        if C1_prime * C2_prime == 0:
            delta_h = 0
        elif abs(h2_prime - h1_prime) <= 180:
            delta_h = h2_prime - h1_prime
        elif h2_prime - h1_prime > 180:
            delta_h = h2_prime - h1_prime - 360
        else:
            delta_h = h2_prime - h1_prime + 360

        delta_H = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(delta_h / 2))

        # Calculate averages
        L_avg = (L1 + L2) / 2
        C_avg_prime = (C1_prime + C2_prime) / 2

        if C1_prime * C2_prime == 0:
            h_avg = h1_prime + h2_prime
        elif abs(h2_prime - h1_prime) <= 180:
            h_avg = (h1_prime + h2_prime) / 2
        elif h1_prime + h2_prime < 360:
            h_avg = (h1_prime + h2_prime + 360) / 2
        else:
            h_avg = (h1_prime + h2_prime - 360) / 2

        T = (1 - 0.17 * np.cos(np.radians(h_avg - 30))
             + 0.24 * np.cos(np.radians(2 * h_avg))
             + 0.32 * np.cos(np.radians(3 * h_avg + 6))
             - 0.20 * np.cos(np.radians(4 * h_avg - 63)))

        delta_theta = 30 * np.exp(-((h_avg - 275) / 25)**2)
        R_C = 2 * np.sqrt(C_avg_prime**7 / (C_avg_prime**7 + 25**7))
        S_L = 1 + (0.015 * (L_avg - 50)**2) / np.sqrt(20 + (L_avg - 50)**2)
        S_C = 1 + 0.045 * C_avg_prime
        S_H = 1 + 0.015 * C_avg_prime * T
        R_T = -np.sin(np.radians(2 * delta_theta)) * R_C

        delta_E = np.sqrt(
            (delta_L / S_L)**2
            + (delta_C / S_C)**2
            + (delta_H / S_H)**2
            + R_T * (delta_C / S_C) * (delta_H / S_H)
        )

        return delta_E

    def is_likely_skin(self, rgb: Tuple[int, int, int]) -> bool:
        """
        Detect if a color is likely a skin tone.

        Uses YCrCb color space thresholds derived from skin detection research.

        Args:
            rgb: RGB tuple (0-255 range)

        Returns:
            True if color is likely a skin tone
        """
        r, g, b = rgb

        # Convert to YCrCb
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cr = (r - y) * 0.713 + 128
        cb = (b - y) * 0.564 + 128

        # Skin detection thresholds (Chai & Ngan, 1999)
        # Adjusted for tattoo analysis context
        skin_cr_min, skin_cr_max = 135, 180
        skin_cb_min, skin_cb_max = 85, 135

        is_skin = (skin_cr_min <= cr <= skin_cr_max and
                   skin_cb_min <= cb <= skin_cb_max)

        # Additional check for very dark or very light colors
        # which are less likely to be skin in tattoo context
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        if luminance < 0.15 or luminance > 0.85:
            is_skin = False

        return is_skin

    def classify(self, rgb: Tuple[int, int, int]) -> Dict[str, any]:
        """
        Classify a single RGB color into a tattoo ink category.

        Args:
            rgb: RGB tuple (0-255 range)

        Returns:
            Dictionary containing:
                - ink_type: Specific ink classification
                - category: Simplified category
                - confidence: Classification confidence (0-1)
                - is_skin: Whether color appears to be skin
        """
        rgb_array = np.array(rgb)
        query_lab = self._rgb_to_lab(rgb_array)

        # Calculate distance to all reference colors
        distances = {}
        for name, ref_lab in self._reference_lab.items():
            distances[name] = self._delta_e_2000(query_lab, ref_lab)

        # Find closest match
        closest = min(distances, key=distances.get)
        min_distance = distances[closest]

        # Calculate confidence (inverse of distance, normalized)
        # Delta E < 1: imperceptible, < 5: noticeable, > 10: very different
        confidence = max(0, min(1, 1 - (min_distance / 50)))

        return {
            "ink_type": closest,
            "category": self.CATEGORY_MAP[closest],
            "confidence": round(confidence, 3),
            "is_skin": self.is_likely_skin(rgb),
            "delta_e": round(min_distance, 2),
        }

    def classify_batch(
        self,
        colors: List[Tuple[int, int, int]],
        filter_skin: bool = True
    ) -> List[Dict[str, any]]:
        """
        Classify multiple colors at once.

        Args:
            colors: List of RGB tuples
            filter_skin: If True, mark skin tones but include them

        Returns:
            List of classification dictionaries
        """
        results = []
        for rgb in colors:
            result = self.classify(rgb)
            results.append(result)
        return results
