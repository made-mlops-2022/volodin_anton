from setuptools import find_packages, setup


setup(
    name="ml-project",
    packages=find_packages(),
    version="0.1.0",
    description="MADE MLOps homework 1",
    author="Anton Volodin",
    license="",
    entry_points={
        "console_scripts": [
            "train = src.models.train_model:train_model",
            "predict = src.models.predict_model:predict_model",
            "make_dataset = src.data.make_dataset:make_dataset",
        ],
    },
    install_requires=[
        "hydra-core",
        "sklearn",
        "pandas",
        "numpy",
        "sweetviz",
    ],
)
