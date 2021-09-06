from setuptools import setup, find_packages

__version__ = "1.0"
setup(
    name="question-answering",
    version=__version__,
    python_requires="~=3.7",
    install_requires=[
        "numpy==1.21.2",  # fix caffe2 ImportError: No module named 'numpy.core._multiarray_umath'
        "optuna==2.9.1",
        "pandas==1.3.2",
        "pyarrow==5.0.0",
        "scikit-learn==0.24.2",
        "pytorch-lightning==1.4.5",
        "transformers==4.10.0",
        "tqdm==4.62.2",
    ],
    extras_require={
        "tests": [
            "black==21.8b0",
            "mypy==0.910",
            "pytest==6.2.5",
            "pytest-cov==2.12.1",
        ],
        "notebook": ["jupyterlab==1.2.16", "ipywidgets==7.6.3", "seaborn==0.11.1"],
    },
    packages=find_packages("src", exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    package_dir={"": "src"},
    include_package_data=True,
    description="Question answering",
    license="MIT",
    author="seahrh",
    author_email="seahrh@gmail.com",
    url="https://github.com/seahrh/question-answering",
)
