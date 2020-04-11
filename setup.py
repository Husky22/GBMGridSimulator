import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GBMGridSimulator-pkg-niklas", # Replace with your own username
    version="0.0.1",
    author="Niklas von Minckwitz",
    author_email="niklas.v.minckwitz@gmail.com",
    description="A threeML GBM grid wrapper.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Husky22/GBMGridSimulator",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
