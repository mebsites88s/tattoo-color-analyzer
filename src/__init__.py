"""
Tattoo Color Analyzer

A tool for analyzing tattoo pigment colors to estimate laser removal difficulty.
Developed by Think Again Tattoo Removal.

https://thinkagaintattooremoval.com
"""

__version__ = "1.0.0"
__author__ = "Think Again Tattoo Removal"

from .analyzer import TattooAnalyzer
from .color_classifier import ColorClassifier
from .difficulty_scorer import DifficultyScorer

__all__ = ["TattooAnalyzer", "ColorClassifier", "DifficultyScorer"]
