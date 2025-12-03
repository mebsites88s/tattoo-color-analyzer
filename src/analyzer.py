"""
Tattoo Color Analyzer

Main analysis module for extracting and analyzing tattoo pigment colors
from photographs using computer vision techniques.

Process:
1. Load and preprocess image
2. Optionally mask non-tattoo regions
3. Extract dominant colors via K-means clustering
4. Classify colors into ink categories
5. Calculate removal difficulty scores
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.cluster import KMeans
from collections import Counter

from .color_classifier import ColorClassifier
from .difficulty_scorer import DifficultyScorer


class TattooAnalyzer:
    """
    Analyzes tattoo images to extract pigment colors and estimate
    removal difficulty.
    """

    def __init__(
        self,
        n_colors: int = 8,
        fitzpatrick_type: Optional[int] = None,
        min_color_percentage: float = 0.02
    ):
        """
        Initialize the tattoo analyzer.

        Args:
            n_colors: Number of dominant colors to extract (default: 8)
            fitzpatrick_type: Patient skin type (1-6) for difficulty calculation
            min_color_percentage: Minimum percentage threshold for color inclusion
        """
        self.n_colors = n_colors
        self.fitzpatrick_type = fitzpatrick_type
        self.min_color_percentage = min_color_percentage
        
        self.classifier = ColorClassifier()
        self.scorer = DifficultyScorer(fitzpatrick_type)
        
        self._image = None
        self._processed_image = None
        self._dominant_colors = None

    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load and validate an image file.

        Args:
            image_path: Path to image file (jpg, png, etc.)

        Returns:
            Loaded image as numpy array (BGR format)

        Raises:
            FileNotFoundError: If image path doesn't exist
            ValueError: If file is not a valid image
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not load image: {path}")

        self._image = image
        return image

    def preprocess(
        self,
        image: Optional[np.ndarray] = None,
        resize_max: int = 800,
        denoise: bool = True
    ) -> np.ndarray:
        """
        Preprocess image for color extraction.

        Args:
            image: Input image (uses loaded image if None)
            resize_max: Maximum dimension for resizing
            denoise: Apply denoising filter

        Returns:
            Preprocessed image
        """
        if image is None:
            if self._image is None:
                raise ValueError("No image loaded")
            image = self._image.copy()

        # Resize if too large (speeds up processing)
        h, w = image.shape[:2]
        if max(h, w) > resize_max:
            scale = resize_max / max(h, w)
            image = cv2.resize(image, None, fx=scale, fy=scale)

        # Optional denoising
        if denoise:
            image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

        self._processed_image = image
        return image

    def extract_dominant_colors(
        self,
        image: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None
    ) -> List[Dict[str, any]]:
        """
        Extract dominant colors using K-means clustering.

        Args:
            image: Input image (uses preprocessed image if None)
            mask: Binary mask to limit analysis region (255 = include)

        Returns:
            List of dominant color dictionaries with RGB, hex, and percentage
        """
        if image is None:
            image = self._processed_image
            if image is None:
                if self._image is not None:
                    image = self.preprocess()
                else:
                    raise ValueError("No image available")

        # Convert to RGB for consistent output
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply mask if provided
        if mask is not None:
            pixels = image_rgb[mask > 0]
        else:
            pixels = image_rgb.reshape(-1, 3)

        # Filter out very bright pixels (likely highlights/reflections)
        brightness = np.mean(pixels, axis=1)
        pixels = pixels[(brightness > 20) & (brightness < 245)]

        if len(pixels) < self.n_colors:
            raise ValueError("Insufficient pixels for color extraction")

        # K-means clustering
        kmeans = KMeans(
            n_clusters=self.n_colors,
            random_state=42,
            n_init=10
        )
        kmeans.fit(pixels)

        # Get cluster centers (dominant colors) and their frequencies
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        label_counts = Counter(labels)
        total_pixels = len(labels)

        # Build color results
        results = []
        for i, color in enumerate(colors):
            percentage = label_counts[i] / total_pixels
            
            if percentage < self.min_color_percentage:
                continue

            rgb = tuple(color.tolist())
            hex_color = "#{:02x}{:02x}{:02x}".format(*rgb)

            results.append({
                "rgb": rgb,
                "hex": hex_color,
                "percentage": round(percentage * 100, 2),
            })

        # Sort by percentage (most dominant first)
        results.sort(key=lambda x: x["percentage"], reverse=True)
        self._dominant_colors = results
        
        return results

    def create_saturation_mask(
        self,
        image: Optional[np.ndarray] = None,
        saturation_threshold: int = 30,
        value_range: Tuple[int, int] = (20, 230)
    ) -> np.ndarray:
        """
        Create a mask to isolate tattooed regions based on color saturation.

        Tattoo ink typically has different saturation characteristics than skin.
        This provides rough segmentation without requiring manual input.

        Args:
            image: Input image (uses loaded image if None)
            saturation_threshold: Minimum saturation to include
            value_range: (min, max) brightness values to include

        Returns:
            Binary mask (255 = potential tattoo region)
        """
        if image is None:
            image = self._image
            if image is None:
                raise ValueError("No image loaded")

        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Create mask based on saturation and value
        mask = np.zeros(s.shape, dtype=np.uint8)
        
        # Include saturated colors (colored inks)
        saturated = s > saturation_threshold
        
        # Include dark regions (black ink)
        dark = v < 80
        
        # Within reasonable brightness range
        valid_brightness = (v >= value_range[0]) & (v <= value_range[1])
        
        # Combine conditions
        mask[(saturated | dark) & valid_brightness] = 255

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    def analyze(
        self,
        image_path: Optional[Union[str, Path]] = None,
        use_mask: bool = True
    ) -> Dict[str, any]:
        """
        Perform complete tattoo analysis.

        Args:
            image_path: Path to image file (uses loaded image if None)
            use_mask: Apply saturation-based masking to isolate tattoo

        Returns:
            Complete analysis results including:
            - dominant_colors: Extracted color data
            - classifications: Ink type classifications
            - difficulty: Removal difficulty assessment
        """
        # Load if path provided
        if image_path is not None:
            self.load_image(image_path)

        # Preprocess
        processed = self.preprocess()

        # Create mask if requested
        mask = None
        if use_mask:
            mask = self.create_saturation_mask(processed)
            # Check if mask captures enough area
            mask_coverage = np.sum(mask > 0) / mask.size
            if mask_coverage < 0.05:
                # Fallback to no mask if too little detected
                mask = None

        # Extract colors
        colors = self.extract_dominant_colors(processed, mask)

        # Classify each color
        classifications = []
        ink_colors = []
        
        for color_data in colors:
            classification = self.classifier.classify(color_data["rgb"])
            
            # Combine with color data
            result = {**color_data, **classification}
            classifications.append(result)
            
            # Track non-skin colors for difficulty scoring
            if not classification["is_skin"]:
                ink_colors.append(result)

        # Build color distribution for scoring (excluding skin)
        color_distribution = {}
        for item in ink_colors:
            category = item["category"]
            if category not in color_distribution:
                color_distribution[category] = 0
            color_distribution[category] += item["percentage"]

        # Calculate difficulty
        if color_distribution:
            difficulty = self.scorer.calculate_composite_score(color_distribution)
        else:
            difficulty = {
                "error": "No ink colors detected",
                "note": "Image may not contain visible tattoo or colors could not be isolated from skin"
            }

        return {
            "dominant_colors": classifications,
            "ink_colors_only": ink_colors,
            "difficulty_assessment": difficulty,
            "analysis_parameters": {
                "n_colors_extracted": self.n_colors,
                "fitzpatrick_type": self.fitzpatrick_type,
                "masking_applied": use_mask and mask is not None,
            }
        }

    def generate_color_palette(
        self,
        output_path: Optional[Union[str, Path]] = None,
        swatch_size: int = 100
    ) -> np.ndarray:
        """
        Generate a visual color palette image.

        Args:
            output_path: Optional path to save palette image
            swatch_size: Size of each color swatch in pixels

        Returns:
            Palette image as numpy array
        """
        if self._dominant_colors is None:
            raise ValueError("No colors extracted. Run analyze() first.")

        colors = self._dominant_colors
        n_colors = len(colors)
        
        # Create palette image
        palette = np.zeros((swatch_size, swatch_size * n_colors, 3), dtype=np.uint8)
        
        for i, color_data in enumerate(colors):
            rgb = color_data["rgb"]
            # OpenCV uses BGR
            bgr = (rgb[2], rgb[1], rgb[0])
            x_start = i * swatch_size
            x_end = (i + 1) * swatch_size
            palette[:, x_start:x_end] = bgr

        if output_path:
            cv2.imwrite(str(output_path), palette)

        return palette


def analyze_image(
    image_path: str,
    n_colors: int = 8,
    fitzpatrick: Optional[int] = None
) -> Dict[str, any]:
    """
    Convenience function for single-image analysis.

    Args:
        image_path: Path to tattoo image
        n_colors: Number of colors to extract
        fitzpatrick: Skin type (1-6)

    Returns:
        Analysis results dictionary
    """
    analyzer = TattooAnalyzer(
        n_colors=n_colors,
        fitzpatrick_type=fitzpatrick
    )
    return analyzer.analyze(image_path)
