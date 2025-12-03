# Example Files

This directory contains sample images and output examples for the Tattoo Color Analyzer.

## Sample Images

Add your own test images here for development and testing.

Recommended test cases:
- Black-only tattoo
- Multicolor tattoo (3+ colors)
- Tattoo with green/turquoise (resistant colors)
- Tattoo on different skin tones

## Sample Output

Example JSON output from analysis:

```json
{
  "dominant_colors": [
    {
      "rgb": [25, 25, 28],
      "hex": "#19191c",
      "percentage": 35.2,
      "ink_type": "black",
      "category": "black",
      "confidence": 0.95,
      "is_skin": false
    },
    {
      "rgb": [45, 85, 160],
      "hex": "#2d55a0",
      "percentage": 22.1,
      "ink_type": "blue",
      "category": "blue",
      "confidence": 0.88,
      "is_skin": false
    }
  ],
  "difficulty_assessment": {
    "composite_difficulty": 4.2,
    "difficulty_label": "Moderate",
    "session_estimate": {
      "minimum": 7,
      "maximum": 14
    },
    "resistant_colors": ["blue"],
    "required_wavelengths": [694, 755, 1064]
  }
}
```

## Usage Examples

```bash
# Basic analysis
tattoo-analyze sample-tattoo.jpg

# With Fitzpatrick skin type
tattoo-analyze sample-tattoo.jpg --fitzpatrick 4

# Export results
tattoo-analyze sample-tattoo.jpg --output analysis.json --palette colors.png
```
