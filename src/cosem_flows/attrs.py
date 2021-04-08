from pathlib import Path
from typing import List, Optional, Union, Literal, Sequence
from xarray_multiscale.metadata.util import SpatialTransform
from pydantic import BaseModel
from enum import Enum

CONTAINER_TYPES = {"mrc", "n5", "precomputed"}
DTYPE_FORMATS = {"uint16": "n5", "uint8": "precomputed", "uint64": "n5"}


class MeshTypeEnum(str, Enum):
    neuroglancer_legacy_mesh = "neuroglancer_legacy_mesh"
    neuroglancer_precomputed_mesh = "neuroglancer_precomputed_mesh"


class ArrayContainerTypeEnum(str, Enum):
    n5 = "n5"
    zarr = "zarr"
    precomputed = "precomputed"
    mrc = "mrc"
    hdf5 = "hdf5"
    tif = "tif"

class ContentTypeEnum(str, Enum):
    em = "em" 
    lm = "lm", 
    prediction = "prediction",
    segmentationm = "segmentation", 
    analysis = "analysis"


class ContrastLimits(BaseModel):
    """
    Metadata for contrast limits. Currently these values are in normalized units, i.e. drawn from the interval [0,1]
    """
    start: float
    end: float
    min: float = 0.0
    max: float = 1.0
    


class DisplaySettings(BaseModel):
    """
    Metadata for display settings
    """
    contrastLimits: ContrastLimits
    color: str = "white"
    invertLUT: bool = False


class DataSource(BaseModel):
    name: str
    path: str
    format: str
    description: str = ""
    version: str = "0"
    transform: SpatialTransform
    tags: Optional[Sequence[str]] = None


class MeshSource(DataSource):
    format: MeshTypeEnum
    ids: Sequence[int]


class VolumeSource(DataSource):
    format: ArrayContainerTypeEnum
    dataType: str
    displaySettings: DisplaySettings
    subsources: Optional[Sequence[MeshSource]]


class DatasetView(BaseModel):
    name: str
    description: str
    position: Optional[Sequence[float]]
    scale: Optional[float]
    orientation: Optional[Sequence[float]]
    volumeIDs: Sequence[str]


class DatasetIndex(BaseModel):
    name: str
    volumes: Sequence[VolumeSource]
    views: Sequence[DatasetView]