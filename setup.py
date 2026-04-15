from setuptools import setup, find_packages

setup(
    name="geoflowkit",
    version="0.1.9",
    packages=find_packages(),
    install_requires=[
        "shapely",
        "numpy",
        "pandas",
        "geopandas>=1.0.1",
        "matplotlib",
        "scikit-learn",
        "tqdm",
        "numba"
    ],
    author="GeoFlow Developer",
    author_email="djw@lreis.ac.cn",
    description="A package for geospatial flow analysis and visualization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/djw-easy/geoflowkit",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT",
    python_requires='>=3.7',
)
