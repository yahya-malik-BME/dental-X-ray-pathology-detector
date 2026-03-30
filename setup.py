from setuptools import setup, find_packages

setup(
    name="dental_xray_cv",
    version="0.1.0",
    description="Dental X-ray pathology detection using YOLOv8 and PyTorch",
    author="yahya-malik-BME",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.2.0",
        "ultralytics>=8.1.0",
        "albumentations>=1.4.0",
        "wandb>=0.16.3",
        "gradio>=4.19.2",
        "hydra-core>=1.3.2",
        "pydantic>=2.6.1",
    ],
)
