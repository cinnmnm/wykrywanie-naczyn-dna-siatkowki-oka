from setuptools import setup, find_packages

setup(
    name="iwm2_projekt",
    version="1.0.0",
    description="Retina vessel Segmentation Project",
    author="Mateusz Juszczak i Anna Zalesińśka",
    packages=find_packages(exclude=["tests*", "legacy*"]),
    install_requires=[
        "torch",
        "torchvision", 
        "numpy",
        "opencv-python",
        "scikit-learn",
        "scikit-image",
        "scipy",
        "pyyaml",
        "matplotlib",
        "joblib"
    ],
    python_requires='>=3.10',
)
