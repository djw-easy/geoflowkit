from setuptools import setup, find_packages

setup(
    name="geoflowkit",
    version="0.3.2",
    packages=find_packages(),
    install_requires=[
        "shapely",
        "numpy",
        "pandas",
        "geopandas>=1.0.1",
        "matplotlib",
        "scipy>=1.9",
        "scikit-learn",
        "tqdm",
        "numba",
        "networkx>=2.6",
    ],
    author="GeoFlow Developer",
    author_email="djw@lreis.ac.cn",
    description="A package for geospatial flow analysis and visualization",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/djw-easy/geoflowkit",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT",
    python_requires='>=3.9',
)
