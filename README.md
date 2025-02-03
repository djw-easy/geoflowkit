GeoFlowKit
---------

Python tools for geographical flow data

Introduction
------------

`geoflowkit` is a project to add support for geographical flow data to
[geopandas](https://geopandas.org/) objects.  It currently implements the `FlowSeries` and `FlowDataFrame` types, 
which are subclasses of `pandas.Series` and `pandas.DataFrame` respectively, 
similar to `geopandas.GeoSeries` and `geopandas.FlowDataFrame`. 
`geoflowkit` objects can act on [shapely](http://shapely.readthedocs.io/en/latest/)
geometry objects and perform geographical flow operations.

Notes:

    - Most functionality is adapted from GeoDataFrame and GeoSeries. 
    - Direct inheritance from GeoDataFrame is avoided due to some unique properties of flow. 
    - Supports all standard pandas DataFrame operations and some geopandas GeoDataFrame operations.
    - Adds geographic operations specific to flow data

