import sys
import os
import warnings
from pathlib import Path


import numpy as np
import pandas as pd
import geopandas as gpd
from pandas import Series
from geopandas import GeoSeries
from geopandas.array import GeometryArray

from flow import Flow


class FlowSeries(GeoSeries):
    """
    A Series object designed to store Flow objects exclusively.
    All data in this series must be instances of Flow class.
    """
    
    _metadata = ['name']
    
    @property
    def _constructor(self):
        return FlowSeries
        
    @property
    def _constructor_sliced(self):
        return FlowSeries
        
    @property
    def _constructor_expanddim(self):
        from geoflow.flowdataframe import FlowDataFrame
        return FlowDataFrame
        
    def __finalize__(self, other, method=None, **kwargs):
        """Propagate metadata from other to self and ensure FlowSeries type"""
        result = super().__finalize__(other, method=method, **kwargs)
        
        # Convert to FlowSeries immediately
        if not isinstance(result, FlowSeries):
            result = FlowSeries(result)
            
        if isinstance(other, FlowSeries):
            # Copy metadata from other FlowSeries
            for attr in self._metadata:
                if hasattr(other, attr):
                    setattr(result, attr, getattr(other, attr))
        
        # Validate the result
        result._validate()
        return result
        
    @classmethod
    def _concat(cls, to_concat, axis=0):
        """Override concat to ensure proper FlowSeries type preservation"""
        from pandas import concat
        if not all(isinstance(s, FlowSeries) for s in to_concat):
            raise TypeError("Can only concatenate FlowSeries objects")
            
        result = concat(to_concat, axis=axis)
        return FlowSeries(result)
        
    def _validate(self):
        """Validate that all elements are Flow objects"""
        if not all(isinstance(item, Flow) for item in self):
            raise TypeError("All elements must be Flow objects")
        return self
        
    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        """Create a FlowSeries from a sequence of scalars"""
        if not all(isinstance(s, Flow) for s in scalars):
            raise TypeError("All elements must be Flow objects")
        return cls(scalars, dtype=object, copy=copy)
        
    def __array_finalize__(self, obj):
        """Ensure proper type preservation during array operations"""
        if obj is None:
            return
        super().__array_finalize__(obj)
        if isinstance(obj, FlowSeries):
            for attr in self._metadata:
                if hasattr(obj, attr):
                    setattr(self, attr, getattr(obj, attr))
    
    def _validate_data(self, data):
        """Validate that all elements in data are Flow objects"""
        if isinstance(data, pd.Series):
            if not all(isinstance(item, Flow) for item in data):
                raise TypeError("All elements must be Flow objects")
            return data
        elif isinstance(data, (list, np.ndarray, tuple)):
            if not all(isinstance(item, Flow) for item in data):
                raise TypeError("All elements must be Flow objects")
            return data
        elif isinstance(data, Flow):
            return data
        else:
            raise TypeError("Data must be Flow object(s)")

    def __init__(self, data=None, index=None, crs=None, **kwargs):
        name = kwargs.pop("name", None)
        if data is not None:
            # Skip validation if input is already a validated FlowSeries
            if isinstance(data, FlowSeries):
                Series.__init__(data=data, index=index, dtype=object, **kwargs)
                return
            
            if (
                hasattr(data, "crs")
                or (isinstance(data, pd.Series) and hasattr(data.array, "crs"))
            ) and crs:
                data_crs = data.crs if hasattr(data, "crs") else data.array.crs
                if not data_crs:
                    # make a copy to avoid setting CRS to passed GeometryArray
                    data = data.copy()
                else:
                    if not data_crs == crs:
                        raise ValueError(
                            "CRS mismatch between CRS of the passed geometries "
                            "and 'crs'. Use 'GeoSeries.set_crs(crs, "
                            "allow_override=True)' to overwrite CRS or "
                            "'GeoSeries.to_crs(crs)' to reproject geometries. "
                        )
            # Validate data
            data = self._validate_data(data)
            
            if isinstance(data, Flow):
                # fix problem for scalar geometries passed, ensure the list of
                # scalars is of correct length if index is specified
                n = len(index) if index is not None else 1
                data = [data] * n
            
            s = pd.Series(data, index=index, name=name, **kwargs)
            
            index = s.index
            name = s.name
            geometry_array = GeometryArray(np.asarray(data), crs=crs)
        
        # Initialize parent Series with geometry array
        super().__init__(data=geometry_array, index=index, name=name, **kwargs)
        if not self.crs:
            self.crs = crs
    
    def __setitem__(self, key, value):
        """Override to ensure only Flow objects can be set"""
        if not isinstance(value, Flow):
            raise TypeError("Can only set Flow objects")
        super().__setitem__(key, value)
    
    def append(self, other):
        """Override append to ensure type safety and type preservation"""
        if isinstance(other, FlowSeries):
            return self._concat([self, other])
        elif isinstance(other, Flow):
            return self._concat([self, FlowSeries([other])])
        else:
            raise TypeError("Can only append Flow objects or FlowSeries")
    
    def _validate_flow_type(self, value):
        """Validate that the value is a Flow object"""
        if not isinstance(value, Flow):
            raise TypeError(f"Value must be Flow object, got {type(value)}")
        return value
    
    def astype(self, dtype, copy=True, errors='raise'):
        """Override astype to prevent type conversion"""
        if dtype != object:
            raise TypeError("FlowSeries can only have dtype object")
        return self.copy() if copy else self


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from geoflow.flow import Flow
    
    # Create test Flow objects
    flow1 = Flow([[0, 0], [1, 1]])
    flow2 = Flow([[1, 1], [2, 2]])
    flow3 = Flow([[2, 2], [3, 3]])
    
    # Test 1: Create FlowSeries with Flow objects
    print("Test 1: Creating FlowSeries with Flow objects")
    fs = FlowSeries([flow1, flow2])
    print(f"FlowSeries created successfully with {len(fs)} flows")
    
    # Test 2: Try to add non-Flow object (should raise TypeError)
    print("\nTest 2: Adding non-Flow object")
    try:
        fs[2] = [0, 1]  # This should raise TypeError
    except TypeError as e:
        print(f"Successfully caught error: {e}")
    
    # Test 3: Append another Flow
    print("\nTest 3: Appending Flow object")
    fs = fs.append(FlowSeries([flow3]))
    assert isinstance(fs, FlowSeries), "Appended object is not a FlowSeries"
    print(f"Successfully appended flow, new length: {len(fs)}")
    
    # Test 4: Try type conversion (should raise TypeError)
    print("\nTest 4: Attempting type conversion")
    try:
        fs = fs.astype(float)  # This should raise TypeError
    except TypeError as e:
        print(f"Successfully caught error: {e}")
        
    # Test 5: Try to use FlowSeries._concat
    print("\nTest 5: Using FlowSeries._concat")
    fs2 = FlowSeries([flow2, flow3])
    try:
        fs3 = FlowSeries._concat([fs, fs2])  # Use our custom concat
        assert isinstance(fs3, FlowSeries), "Result is not a FlowSeries"
        print("Successfully concatenated FlowSeries")
    except Exception as e:
        print(f"Error concatenating FlowSeries: {e}")
        
    # Test 6: Try to use FlowSeries._concat with non-FlowSeries
    print("\nTest 6: Using FlowSeries._concat with non-FlowSeries")
    s3 = pd.Series([1, 2, 3])
    try:
        fs3 = FlowSeries._concat([fs, s3])  # This should raise TypeError
    except TypeError as e:
        print(f"Successfully caught error: {e}")
    
    print("\nAll tests completed!")