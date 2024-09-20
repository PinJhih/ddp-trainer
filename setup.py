from setuptools import setup, find_packages

setup(
    name="ddp_trainer",
    version="0.0.1",
    description="A description of your package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="PinJhih Wang",
    author_email="acs110134@gm.ntcu.edu.tw",
    url="https://github.com/PinJhih/ddp-trainer",
    license="MIT",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "numpy",
        "matplotlib",
    ],
)
