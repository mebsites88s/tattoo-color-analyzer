from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tattoo-color-analyzer",
    version="1.0.0",
    author="Think Again Tattoo Removal",
    author_email="info@thinkagaintattooremoval.com",
    description="Analyze tattoo photos to estimate laser removal difficulty based on pigment colors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thinkagaintattooremoval/tattoo-color-analyzer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "Pillow>=10.0.0",
    ],
    entry_points={
        "console_scripts": [
            "tattoo-analyze=src.cli:main",
        ],
    },
)
