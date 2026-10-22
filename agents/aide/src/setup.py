from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
long_description = (ROOT / "README.md").read_text()


def read_requirements(path):
    """Expand local groups so pip -r and package installation use the same set."""
    requirements = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r "):
            requirements.extend(read_requirements(path.parent / line[3:].strip()))
        else:
            requirements.append(line)
    return list(dict.fromkeys(requirements))


requirements = read_requirements(ROOT / "requirements.txt")

setup(
    name="aideml",
    version="0.1.4",
    author="Weco AI",
    author_email="contact@weco.ai",
    description="Autonomous AI for Data Science and Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Wecoai/aideml",
    packages=find_packages(),
    package_data={
        "aide": [
            "../requirements*.txt",
            "utils/config.yaml",
            "utils/viz_templates/*",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "aide = aide.run:run",
        ],
    },
)
