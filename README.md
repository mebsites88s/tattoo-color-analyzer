# Tattoo Color Analyzer

A Python CLI tool for analyzing tattoo photographs to extract pigment colors and estimate laser removal difficulty. Built for clinical consultation preparation and patient education.

Developed by [Think Again Tattoo Removal](https://thinkagaintattooremoval.com) - Medical Laser Tattoo Removal.

## Features

- **Color Extraction**: Identifies dominant pigment colors using K-means clustering
- **Ink Classification**: Categorizes colors into standard tattoo ink types (black, blue, green, red, yellow, white, etc.)
- **Difficulty Scoring**: Calculates removal difficulty based on clinical evidence
- **Session Estimation**: Provides estimated session ranges based on color composition
- **Skin Type Consideration**: Adjusts estimates for Fitzpatrick skin phototypes I-VI
- **Skin Tone Filtering**: Automatically distinguishes ink colors from background skin

## Installation

```bash
# Clone repository
git clone https://github.com/thinkagaintattooremoval/tattoo-color-analyzer.git
cd tattoo-color-analyzer

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Requirements

- Python 3.8+
- OpenCV
- NumPy
- scikit-learn
- Pillow

## Usage

### Command Line

```bash
# Basic analysis
tattoo-analyze image.jpg

# Specify number of colors to extract
tattoo-analyze image.jpg --colors 10

# Include Fitzpatrick skin type for adjusted estimates
tattoo-analyze image.jpg --fitzpatrick 4

# Export results
tattoo-analyze image.jpg --output results.json --palette palette.png

# JSON output for integration
tattoo-analyze image.jpg --json
```

### Python API

```python
from src import TattooAnalyzer

# Initialize analyzer
analyzer = TattooAnalyzer(n_colors=8, fitzpatrick_type=3)

# Analyze image
results = analyzer.analyze("tattoo.jpg")

# Access results
for color in results["dominant_colors"]:
    print(f"{color['hex']} - {color['category']} ({color['percentage']:.1f}%)")

# Difficulty assessment
difficulty = results["difficulty_assessment"]
print(f"Difficulty: {difficulty['composite_difficulty']}/10")
print(f"Sessions: {difficulty['session_estimate']['minimum']}-{difficulty['session_estimate']['maximum']}")
```

## Methodology

### Color Extraction

The analyzer uses K-means clustering on preprocessed image pixels to identify dominant colors. An optional saturation-based mask isolates tattooed regions from surrounding skin.

### Color Classification

Extracted RGB values are converted to CIELAB color space for perceptually uniform distance calculations. The CIEDE2000 formula (the most accurate color difference metric) determines the closest match to reference ink colors.

### Difficulty Scoring

Removal difficulty scores are based on established clinical literature regarding laser-ink interactions:

| Color | Base Difficulty | Notes |
|-------|-----------------|-------|
| Black | 2.0 | Absorbs all wavelengths; responds to most Q-switched lasers |
| Red | 4.0 | Responds to 532nm frequency-doubled Nd:YAG |
| Brown | 4.5 | Iron oxide pigments; moderate response |
| Purple | 5.5 | Mixed chromophores; may require multiple wavelengths |
| Blue | 6.0 | Requires 694nm ruby or 755nm alexandrite |
| Yellow | 7.5 | Light pigment reflects laser energy |
| Green | 8.0 | Most resistant; often requires multiple laser types |
| Turquoise | 8.5 | Blue-green extremely resistant |
| White | 9.0 | Titanium dioxide may paradoxically darken |

### Session Estimation

Base session estimates (6-12 for black ink) are modified by:
- Color-specific multipliers (green = 2.0x, blue = 1.5x, etc.)
- Fitzpatrick skin type adjustments (Types V-VI require extended intervals)
- Color complexity (multiple resistant colors add additional sessions)

### Fitzpatrick Skin Type Considerations

Higher melanin content increases risk of:
- Post-inflammatory hyperpigmentation (PIH)
- Post-inflammatory hypopigmentation
- Prolonged healing time

Patients with Fitzpatrick Types IV-VI typically require:
- Lower fluence settings
- Extended treatment intervals (8-12 weeks)
- Preference for 1064nm Nd:YAG wavelength

## Clinical References

The methodology and difficulty scoring are informed by peer-reviewed literature:

1. Bernstein, E. F. (2017). Laser tattoo removal. *Seminars in Plastic Surgery, 31*(3), 163-170. https://doi.org/10.1055/s-0037-1604297

2. Ho, S. G., & Goh, C. L. (2015). Laser tattoo removal: A clinical update. *Journal of Cutaneous and Aesthetic Surgery, 8*(1), 9-15. https://doi.org/10.4103/0974-2077.155066

3. Kirby, W., Desai, A., Desai, T., Kartono, F., & Geeta, P. (2010). The Kirby-Desai Scale: A proposed scale to assess tattoo-removal treatments. *Journal of Clinical and Aesthetic Dermatology, 3*(2), 32-37.

4. Fitzpatrick, T. B. (1988). The validity and practicality of sun-reactive skin types I through VI. *Archives of Dermatology, 124*(6), 869-871.

5. Kuperman-Beade, M., Levine, V. J., & Ashinoff, R. (2001). Laser removal of tattoos. *American Journal of Clinical Dermatology, 2*(1), 21-25.

## Example Output

```
============================================================
TATTOO COLOR ANALYSIS RESULTS
============================================================

DOMINANT COLORS DETECTED:
----------------------------------------
  1. #1a1a1c - BLACK (42.3%)
  2. #2d5590 - BLUE (28.1%)
  3. #c4a590 - BROWN (15.2%) [SKIN]
  4. #258040 - GREEN (9.8%)
  5. #d04535 - RED (4.6%)

INK COLORS (excluding skin tones):
----------------------------------------
  BLACK: 42.3%
  BLUE: 28.1%
  GREEN: 9.8%
  RED: 4.6%

REMOVAL DIFFICULTY ASSESSMENT:
----------------------------------------
  Overall Difficulty: 5.8/10 (Challenging)
  Estimated Sessions: 10-18
  Resistant Colors: blue, green
  Suggested Wavelengths: 532, 694, 755, 1064nm
  Fitzpatrick Type: 2

============================================================
Developed by Think Again Tattoo Removal
https://thinkagaintattooremoval.com
============================================================
```

## About Think Again Tattoo Removal

Think Again Tattoo Removal is a medical laser tattoo removal provider operating in the United States. Our practice is physician-supervised, utilizing advanced Q-switched and picosecond laser technology for safe, effective tattoo removal across all skin types.

**Website**: [https://thinkagaintattooremoval.com](https://thinkagaintattooremoval.com)

## Medical Disclaimer

**This tool is for educational and informational purposes only.** The color analysis and session estimates provided are algorithmic approximations and do not constitute medical advice, diagnosis, or treatment recommendations.

**Important considerations not captured by image analysis include:**

- Ink depth and density
- Tattoo age and application technique
- Anatomical location (distal extremities heal slower)
- Patient healing response and immune function
- Presence of scarring or layered tattoos
- Specific ink formulations and carrier agents
- Treatment protocol and laser parameters

**Actual removal outcomes vary significantly between patients.** Consultation with a qualified medical provider experienced in laser tattoo removal is essential for accurate assessment and treatment planning.

This tool does not establish a patient-provider relationship. Always seek evaluation from a licensed healthcare professional before pursuing laser tattoo removal treatment.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome. Please open an issue to discuss proposed changes before submitting a pull request.
