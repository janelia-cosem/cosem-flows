from typing import Any, Callable, Dict, Tuple, Optional, List, TypeVar, Sequence
from fibsem_tools.io.io import access_precomputed, read_xarray, split_path_at_suffix
import numpy as np
from pydantic.types import DirectoryPath
import xarray
from xarray_multiscale.multiscale import multiscale, get_downscale_depth
from fibsem_tools.metadata.cosem import COSEMGroupMetadata, SpatialTransform, ScaleMeta
from fibsem_tools.metadata.neuroglancer import NeuroglancerN5GroupMetadata
from xarray_multiscale.reducers import mode
import tensorstore as ts
from xarray import DataArray

from fibsem_tools.io import initialize_group
from fibsem_tools.io.dask import store_blocks
import click
import dask.array as da




class Multiscales():
    def __init__(self, name: str, arrays: Dict[str, DataArray], attrs: Dict[str, Any]={}):
        if not isinstance(arrays, dict):
            raise ValueError('arrays must be a dict of xarray.DataArrays')
        else:
            if not all(isinstance(x, xarray.DataArray) for x in arrays.values()):
                raise ValueError('arrays must be a dict of xarray.DataArrays')
        self.arrays: Dict[str, DataArray] = arrays
        self.attrs = attrs
        self.name = name
        
    def __repr__(self):
        return str(self.arrays)
    
   
    def store(self, store: str, output_chunks: Optional[Tuple[int]]=None, multiscale_metadata: bool=True, propagate_array_attrs:bool=True, **kwargs):
        """
        Prepare to store the multiscale arrays.


        """
        
        group_attrs = {**self.attrs}        
        array_attrs = {k: {} for k in self.arrays}

        if propagate_array_attrs: 
            array_attrs = {k: dict(v.attrs) for k, v in self.arrays.items()}

        if multiscale_metadata:
            group_attrs.update(COSEMGroupMetadata.fromDataArrays(name=self.name, dataarrays=tuple(self.arrays.values()), paths=tuple(self.arrays.keys())).dict())
            group_attrs.update(NeuroglancerN5GroupMetadata.fromDataArrays(self.arrays.values()).dict())

            for k, arr in self.arrays.items():
                array_attrs[k].update(ScaleMeta(path=k, transform=SpatialTransform.fromDataArray(arr)).dict())

        
        if output_chunks is None:
            _output_chunks = [v.data.chunksize for v in self.arrays.values()]
        else:
            _output_chunks = (output_chunks,) * len(self.arrays.values())

        store_group, store_arrays = initialize_group(
            store,
            self.name,
            tuple(self.arrays.values()),
            array_paths=tuple(self.arrays.keys()),
            chunks=_output_chunks,
            group_attrs=group_attrs,
            array_attrs=tuple(array_attrs.values()),
            **kwargs
        )
                   
        storage_ops = store_blocks([v.data for v in self.arrays.values()], store_arrays)
        return store_group, store_arrays, storage_ops