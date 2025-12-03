#!/usr/bin/env python3
"""
Tattoo Color Analyzer CLI

Command-line interface for analyzing tattoo images and estimating
removal difficulty.

Usage:
    tattoo-analyze image.jpg
    tattoo-analyze image.png --colors 10 --fitzpatrick 3
    tattoo-analyze image.jpg --output results.json --palette palette.png
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .analyzer import TattooAnalyzer


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        prog="tattoo-analyze",
        description=(
            "Analyze tattoo photographs to extract pigment colors and "
            "estimate laser removal difficulty. Developed by Think Again "
            "Tattoo Removal (thinkagaintattooremoval.com)."
        ),
        epilog=(
            "DISCLAIMER: This tool provides estimates for educational purposes "
            "only. Actual removal outcomes depend on many factors including ink "
            "depth, density, patient healing, and treatment protocols. Consult "
            "with a qualified medical provider for accurate assessment."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "image",
        type=str,
        help="Path to tattoo image file (JPG, PNG)"
    )

    parser.add_argument(
        "-c", "--colors",
        type=int,
        default=8,
        help="Number of dominant colors to extract (default: 8)"
    )

    parser.add_argument(
        "-f", "--fitzpatrick",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        default=None,
        help="Fitzpatrick skin type (1-6) for adjusted difficulty scoring"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Save JSON results to file"
    )

    parser.add_argument(
        "-p", "--palette",
        type=str,
        default=None,
        help="Generate and save color palette image"
    )

    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Disable automatic tattoo region masking"
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress console output (use with --output)"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON to stdout"
    )

    parser.add_argument(
        "-v", "--version",
        action="version",
        version="%(prog)s 1.0.0"
    )

    return parser


def format_results(results: dict) -> str:
    """Format analysis results for console output."""
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("TATTOO COLOR ANALYSIS RESULTS")
    lines.append("=" * 60)

    # Dominant colors
    lines.append("\nDOMINANT COLORS DETECTED:")
    lines.append("-" * 40)
    
    for i, color in enumerate(results["dominant_colors"], 1):
        skin_marker = " [SKIN]" if color.get("is_skin") else ""
        lines.append(
            f"  {i}. {color['hex']} - {color['category'].upper()}"
            f" ({color['percentage']:.1f}%){skin_marker}"
        )

    # Ink colors summary
    ink_only = results.get("ink_colors_only", [])
    if ink_only:
        lines.append("\nINK COLORS (excluding skin tones):")
        lines.append("-" * 40)
        
        categories = {}
        for color in ink_only:
            cat = color["category"]
            if cat not in categories:
                categories[cat] = 0
            categories[cat] += color["percentage"]
        
        for cat, pct in sorted(categories.items(), key=lambda x: -x[1]):
            lines.append(f"  {cat.upper()}: {pct:.1f}%")

    # Difficulty assessment
    difficulty = results.get("difficulty_assessment", {})
    if "error" not in difficulty:
        lines.append("\nREMOVAL DIFFICULTY ASSESSMENT:")
        lines.append("-" * 40)
        lines.append(
            f"  Overall Difficulty: {difficulty['composite_difficulty']}/10 "
            f"({difficulty['difficulty_label']})"
        )
        
        sessions = difficulty.get("session_estimate", {})
        lines.append(
            f"  Estimated Sessions: {sessions.get('minimum', '?')}-"
            f"{sessions.get('maximum', '?')}"
        )
        
        resistant = difficulty.get("resistant_colors", [])
        if resistant:
            lines.append(f"  Resistant Colors: {', '.join(resistant)}")
        
        wavelengths = difficulty.get("required_wavelengths", [])
        if wavelengths:
            lines.append(f"  Suggested Wavelengths: {', '.join(map(str, wavelengths))}nm")
        
        fitz = difficulty.get("fitzpatrick_type")
        if fitz:
            lines.append(f"  Fitzpatrick Type: {fitz}")

        considerations = difficulty.get("skin_considerations", [])
        if considerations and considerations != ["Standard treatment protocols apply"]:
            lines.append("\n  SKIN TYPE CONSIDERATIONS:")
            for note in considerations:
                lines.append(f"    - {note}")

    lines.append("\n" + "=" * 60)
    lines.append("Developed by Think Again Tattoo Removal")
    lines.append("https://thinkagaintattooremoval.com")
    lines.append("=" * 60 + "\n")

    return "\n".join(lines)


def main(args: Optional[list] = None) -> int:
    """
    Main CLI entry point.

    Args:
        args: Command line arguments (uses sys.argv if None)

    Returns:
        Exit code (0 = success, 1 = error)
    """
    parser = create_parser()
    opts = parser.parse_args(args)

    # Validate image path
    image_path = Path(opts.image)
    if not image_path.exists():
        print(f"Error: Image not found: {opts.image}", file=sys.stderr)
        return 1

    if not image_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
        print(f"Warning: Unexpected file type: {image_path.suffix}", file=sys.stderr)

    try:
        # Initialize analyzer
        analyzer = TattooAnalyzer(
            n_colors=opts.colors,
            fitzpatrick_type=opts.fitzpatrick
        )

        # Run analysis
        results = analyzer.analyze(
            image_path=image_path,
            use_mask=not opts.no_mask
        )

        # Generate palette if requested
        if opts.palette:
            palette_path = Path(opts.palette)
            analyzer.generate_color_palette(palette_path)
            if not opts.quiet:
                print(f"Color palette saved to: {palette_path}")

        # Save JSON if requested
        if opts.output:
            output_path = Path(opts.output)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            if not opts.quiet:
                print(f"Results saved to: {output_path}")

        # Console output
        if opts.json:
            print(json.dumps(results, indent=2))
        elif not opts.quiet:
            print(format_results(results))

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
