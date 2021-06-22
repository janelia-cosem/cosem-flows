from .save_multiscale import typed_list_from_mongodb
from fibsem_metadata.index import DatasetView, VolumeSource, DatasetIndex, MeshSource, DatasetViewCollection
from pathlib import Path
import fsspec
from typing import List
from dataclasses import replace
import os
import json
import warnings


def get_neuroglancer_legacy_mesh_ids(path: str):
    """
    Given a path to a directory with files called 1.ngmesh, 2.ngmesh, etc, return (1,2,...)
    """
    protocol = fsspec.utils.get_protocol(path)
    fs = fsspec.filesystem(protocol)
    return sorted(
        tuple(
            map(
                lambda v: int(v.split(".")[0].split("/")[-1]),
                fs.glob(os.path.join(path, "*.ngmesh")),
            )
        )
    )


def infer_container_type(path: str) -> str:
    if Path(path).suffix == ".mrc":
        containerType = "mrc"
    elif any(map(lambda v: (Path(v).suffix == ".n5"), Path(path).parts)):
        containerType = "n5"
    elif any(map(lambda v: (Path(v).suffix == ".precomputed"), Path(path).parts)):
        containerType = "precomputed"
    else:
        raise ValueError(f"Could not infer container type from path {path}")

    return containerType

def URLify(protocol: str, path: str):
    return protocol + '://' + path


def build_index(URL: str, volume_registry: str):
    try:
        fs_protocol, root = URL.split("://")
    except ValueError:
        print(f"Your input {URL} could not be split into a protocol and a path")
        raise
    fs = fsspec.filesystem(fs_protocol)
    dataset_name = Path(root).name
    registry_mapper = fsspec.get_mapper(os.path.join(volume_registry, dataset_name))
    n5_paths = tuple(filter(fs.isdir, fs.glob(os.path.join(root, "*n5*/*/*"))))
    precomputed_paths = tuple(filter(fs.isdir, fs.glob(os.path.join(root, "neuroglancer/em/*"))))
    mesh_paths = {Path(p).stem: p for p in  filter(fs.isdir, fs.glob(os.path.join(root, "neuroglancer/mesh/*")))}
    volume_paths = {Path(p).stem: p for p in (*n5_paths, *precomputed_paths)}
    registry_volumes = []
    missing_from_registry = []
    for stem in volume_paths:
        pth = os.path.join('sources', stem + '.json')
        try:
            json_payload = json.loads(registry_mapper[pth])
            registry_volumes.append(VolumeSource(**json_payload))
        except KeyError:
            missing_from_registry.append(volume_paths[stem])
    output_volume_sources: List[VolumeSource] = []
    for vj in registry_volumes:
        vol_path = volume_paths[vj.name]
        vj.path = URLify(fs_protocol, vol_path)
        vj.format = infer_container_type(vol_path)
        subsources = []
        if vj.name in mesh_paths:
            subsources = [MeshSource(name=vj.name, path=URLify(fs_protocol, mesh_paths[vj.name]), transform=vj.transform, format='neuroglancer_legacy_mesh', ids=[])]
        vj.subsources = subsources
        output_volume_sources.append(vj)
    
    db_views: DatasetViewCollection = []    
    db_views = DatasetViewCollection(**json.loads(registry_mapper['views.json'])).views

    accepted_views = []
    rejected_views = []
    for v in db_views:
        missing = set(v.volumeNames) - set(volume_paths.keys())
        if len(missing) > 0:
            rejected_views.append(v)
        else:
            accepted_views.append(v)

    index = DatasetIndex(
        name=dataset_name,
        volumes=output_volume_sources,
        views=accepted_views,
    )

    return index, (registry_volumes, missing_from_registry), (rejected_views, accepted_views)
