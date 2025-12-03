"""
Difficulty Scorer Module

Calculates laser tattoo removal difficulty based on pigment colors
and Fitzpatrick skin type considerations.

Clinical references:
- Bernstein, E. F. (2017). Laser tattoo removal. Seminars in Plastic Surgery, 31(3), 163-170.
- Ho, S. G., & Goh, C. L. (2015). Laser tattoo removal: A clinical update. Journal of Cutaneous and Aesthetic Surgery, 8(1), 9-15.
- Kirby, W., et al. (2010). The Kirby-Desai Scale: A proposed scale to assess tattoo-removal treatments. Journal of Clinical and Aesthetic Dermatology, 3(2), 32-37.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import IntEnum


class FitzpatrickType(IntEnum):
    """
    Fitzpatrick Skin Phototype Classification.
    
    Reference: Fitzpatrick, T. B. (1988). The validity and practicality 
    of sun-reactive skin types I through VI. Archives of Dermatology, 124(6), 869-871.
    """
    TYPE_I = 1    # Very fair, always burns, never tans
    TYPE_II = 2   # Fair, usually burns, tans minimally
    TYPE_III = 3  # Medium, sometimes burns, tans uniformly
    TYPE_IV = 4   # Olive, rarely burns, tans well
    TYPE_V = 5    # Brown, very rarely burns, tans very easily
    TYPE_VI = 6   # Dark brown/black, never burns, deeply pigmented


@dataclass
class ColorDifficulty:
    """Removal difficulty data for a specific color category."""
    category: str
    base_score: float  # 1-10 scale (1 = easiest, 10 = hardest)
    session_multiplier: float  # Factor applied to base session estimate
    optimal_wavelengths: List[int]  # Effective laser wavelengths (nm)
    notes: str


class DifficultyScorer:
    """
    Scores tattoo removal difficulty based on ink colors and skin type.
    
    Difficulty scoring is based on clinical literature examining:
    - Ink particle size and depth
    - Chromophore absorption spectra
    - Laser wavelength effectiveness
    - Skin type risk factors
    """

    # Removal difficulty data by color category
    # Based on Ho & Goh (2015) and Bernstein (2017)
    COLOR_DIFFICULTY: Dict[str, ColorDifficulty] = {
        "black": ColorDifficulty(
            category="black",
            base_score=2.0,
            session_multiplier=1.0,
            optimal_wavelengths=[1064, 755, 694],
            notes="Absorbs all wavelengths; responds to most Q-switched lasers"
        ),
        "blue": ColorDifficulty(
            category="blue",
            base_score=6.0,
            session_multiplier=1.5,
            optimal_wavelengths=[694, 755],
            notes="Requires ruby (694nm) or alexandrite (755nm); resistant to Nd:YAG"
        ),
        "green": ColorDifficulty(
            category="green",
            base_score=8.0,
            session_multiplier=2.0,
            optimal_wavelengths=[694, 755],
            notes="Most resistant; often requires multiple laser types"
        ),
        "turquoise": ColorDifficulty(
            category="turquoise",
            base_score=8.5,
            session_multiplier=2.2,
            optimal_wavelengths=[694, 755],
            notes="Blue-green pigments extremely resistant; may require combination therapy"
        ),
        "red": ColorDifficulty(
            category="red",
            base_score=4.0,
            session_multiplier=1.2,
            optimal_wavelengths=[532],
            notes="Responds to frequency-doubled Nd:YAG (532nm)"
        ),
        "yellow": ColorDifficulty(
            category="yellow",
            base_score=7.5,
            session_multiplier=1.8,
            optimal_wavelengths=[532],
            notes="Light pigment reflects laser energy; limited absorption"
        ),
        "white": ColorDifficulty(
            category="white",
            base_score=9.0,
            session_multiplier=2.5,
            optimal_wavelengths=[],
            notes="Titanium dioxide may paradoxically darken; requires test spot"
        ),
        "purple": ColorDifficulty(
            category="purple",
            base_score=5.5,
            session_multiplier=1.4,
            optimal_wavelengths=[694, 755, 532],
            notes="Mixed red/blue response; may require wavelength combination"
        ),
        "brown": ColorDifficulty(
            category="brown",
            base_score=4.5,
            session_multiplier=1.3,
            optimal_wavelengths=[1064, 755],
            notes="Iron oxide pigments; responds moderately to standard lasers"
        ),
    }

    # Fitzpatrick skin type adjustment factors
    # Higher skin types = increased treatment difficulty and risk
    # Based on Kirby-Desai Scale (2010)
    SKIN_TYPE_FACTORS: Dict[int, Dict[str, float]] = {
        1: {"multiplier": 1.0, "risk_level": "low"},
        2: {"multiplier": 1.0, "risk_level": "low"},
        3: {"multiplier": 1.15, "risk_level": "moderate"},
        4: {"multiplier": 1.3, "risk_level": "moderate"},
        5: {"multiplier": 1.5, "risk_level": "high"},
        6: {"multiplier": 1.7, "risk_level": "high"},
    }

    # Base session estimates (for average black ink tattoo)
    BASE_SESSIONS_MIN = 6
    BASE_SESSIONS_MAX = 12

    def __init__(self, fitzpatrick_type: Optional[int] = None):
        """
        Initialize the difficulty scorer.

        Args:
            fitzpatrick_type: Fitzpatrick skin type (1-6). If None, assumes Type II.
        """
        self.fitzpatrick_type = fitzpatrick_type or 2
        if not 1 <= self.fitzpatrick_type <= 6:
            raise ValueError("Fitzpatrick type must be between 1 and 6")

    def get_skin_factor(self) -> Dict[str, any]:
        """Get adjustment factors for current skin type."""
        return self.SKIN_TYPE_FACTORS[self.fitzpatrick_type]

    def score_color(self, category: str) -> Dict[str, any]:
        """
        Calculate difficulty score for a single color category.

        Args:
            category: Color category name (e.g., "black", "green")

        Returns:
            Dictionary with difficulty metrics
        """
        if category not in self.COLOR_DIFFICULTY:
            # Default to moderate difficulty for unknown colors
            category = "purple"

        color_data = self.COLOR_DIFFICULTY[category]
        skin_factor = self.get_skin_factor()

        # Adjust base score for skin type
        adjusted_score = min(10, color_data.base_score * skin_factor["multiplier"])

        return {
            "category": category,
            "base_difficulty": color_data.base_score,
            "adjusted_difficulty": round(adjusted_score, 1),
            "session_multiplier": color_data.session_multiplier,
            "optimal_wavelengths": color_data.optimal_wavelengths,
            "notes": color_data.notes,
            "skin_risk_level": skin_factor["risk_level"],
        }

    def calculate_composite_score(
        self,
        color_distribution: Dict[str, float]
    ) -> Dict[str, any]:
        """
        Calculate composite difficulty score for multiple colors.

        Uses weighted average based on color prevalence, with additional
        penalty for color complexity (multiple resistant colors).

        Args:
            color_distribution: Dict mapping color categories to their
                               percentage of total tattoo area (0-1)

        Returns:
            Comprehensive difficulty assessment
        """
        if not color_distribution:
            return {"error": "No colors provided"}

        # Normalize distribution
        total = sum(color_distribution.values())
        if total == 0:
            return {"error": "Color distribution sums to zero"}

        normalized = {k: v / total for k, v in color_distribution.items()}

        # Calculate weighted difficulty
        weighted_score = 0
        weighted_multiplier = 0
        all_wavelengths = set()
        resistant_colors = []
        color_details = []

        for category, weight in normalized.items():
            if weight < 0.01:  # Skip negligible colors
                continue

            score_data = self.score_color(category)
            weighted_score += score_data["adjusted_difficulty"] * weight
            weighted_multiplier += score_data["session_multiplier"] * weight
            all_wavelengths.update(score_data["optimal_wavelengths"])

            if score_data["base_difficulty"] >= 6:
                resistant_colors.append(category)

            color_details.append({
                "category": category,
                "percentage": round(weight * 100, 1),
                "difficulty": score_data["adjusted_difficulty"],
            })

        # Complexity penalty: multiple resistant colors add difficulty
        complexity_penalty = 0
        if len(resistant_colors) >= 2:
            complexity_penalty = 0.5 * (len(resistant_colors) - 1)
        
        final_score = min(10, weighted_score + complexity_penalty)

        # Calculate session estimates
        session_min = round(self.BASE_SESSIONS_MIN * weighted_multiplier)
        session_max = round(self.BASE_SESSIONS_MAX * weighted_multiplier)

        # Apply skin type factor to sessions
        skin_factor = self.get_skin_factor()
        if skin_factor["risk_level"] == "high":
            # Longer intervals needed = more calendar time
            session_min = round(session_min * 1.2)
            session_max = round(session_max * 1.3)

        return {
            "composite_difficulty": round(final_score, 1),
            "difficulty_label": self._score_to_label(final_score),
            "session_estimate": {
                "minimum": session_min,
                "maximum": session_max,
                "note": "Actual sessions vary based on ink density, location, and healing"
            },
            "color_breakdown": sorted(
                color_details,
                key=lambda x: x["percentage"],
                reverse=True
            ),
            "resistant_colors": resistant_colors,
            "required_wavelengths": sorted(list(all_wavelengths)),
            "fitzpatrick_type": self.fitzpatrick_type,
            "skin_considerations": self._get_skin_considerations(),
        }

    @staticmethod
    def _score_to_label(score: float) -> str:
        """Convert numeric score to descriptive label."""
        if score <= 3:
            return "Standard"
        elif score <= 5:
            return "Moderate"
        elif score <= 7:
            return "Challenging"
        else:
            return "Complex"

    def _get_skin_considerations(self) -> List[str]:
        """Get treatment considerations for skin type."""
        considerations = []
        
        if self.fitzpatrick_type >= 4:
            considerations.append(
                "Higher risk of post-inflammatory hyperpigmentation; "
                "lower fluence settings recommended"
            )
        if self.fitzpatrick_type >= 5:
            considerations.append(
                "Extended treatment intervals (8-12 weeks) advised to minimize "
                "pigmentary complications"
            )
            considerations.append(
                "Consider Nd:YAG 1064nm as primary wavelength due to reduced "
                "melanin absorption"
            )
        if self.fitzpatrick_type <= 2:
            considerations.append(
                "Lower melanin competition allows for broader wavelength options"
            )

        return considerations if considerations else ["Standard treatment protocols apply"]


def estimate_sessions_from_colors(
    colors: List[str],
    fitzpatrick: int = 2
) -> Tuple[int, int]:
    """
    Quick utility function for session estimation.

    Args:
        colors: List of color categories present in tattoo
        fitzpatrick: Skin type (1-6)

    Returns:
        Tuple of (min_sessions, max_sessions)
    """
    scorer = DifficultyScorer(fitzpatrick)
    
    # Assume equal distribution if just given color list
    distribution = {c: 1.0 / len(colors) for c in colors}
    
    result = scorer.calculate_composite_score(distribution)
    return (
        result["session_estimate"]["minimum"],
        result["session_estimate"]["maximum"]
    )
