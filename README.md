# GeoFlowKit

[![PyPI version](https://badge.fury.io/py/geoflowkit.svg)](https://badge.fury.io/py/geoflowkit)

Python tools for handle geographical flow data. 

Introduction
------------

`geoflowkit` is a project to add support for geographical flow data to
[geopandas](https://geopandas.org/) objects.  It currently implements the `FlowSeries` and `FlowDataFrame` types, 
which are subclasses of `pandas.Series` and `pandas.DataFrame` respectively, 
similar to `geopandas.GeoSeries` and `geopandas.GeoDataFrame`. 
`geoflowkit` objects can act on [shapely](http://shapely.readthedocs.io/en/latest/)
geometry objects and perform geographical flow operations.

Notes:

    - Most functionality is adapted from GeoDataFrame and GeoSeries. 
    - Direct inheritance from GeoDataFrame is avoided due to some unique properties of flow. 
    - Supports all standard pandas DataFrame operations and some geopandas GeoDataFrame operations.
    - Adds geographic operations specific to flow data

Installation
------------

`geoflowkit` requires Python 3.7 or later. It is available on PyPI and can be installed using `pip`:

    pip install geoflowkit

Dependencies:
- geopandas
- numba
- scikit-learn
- matplotlib

Documentation
-------------

Some examples are available in the [examples](examples) directory. For full documentation, please visit our [documentation site](https://geoflowkit.readthedocs.io).

Contributing
------------

We welcome contributions to GeoFlowKit! 

License
-------

GeoFlowKit is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

Contact
-------

For any questions or feedback, please contact: djw@lreis.ac.cn
