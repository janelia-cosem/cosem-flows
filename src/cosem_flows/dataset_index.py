
from .save_multiscale import typed_list_from_mongodb
from fibsem_tools.attrs import DatasetView, VolumeSource, DatasetIndex, MeshSource
from pathlib import Path
from fsspec import filesystem
from typing import List
from dataclasses import replace

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

# mongodb variables
un = "root"
pw = "root"
addr = "cosem.int.janelia.org"
db = "sources"
mongo_addr = f"mongodb://{un}:{pw}@{addr}"

def build_index(URL: str):
    try:
        fs_protocol, root = URL.split('://')
    except ValueError:
        print(f'Your input {URL} could not be split into a protocol and a path')
        raise
    fs = filesystem(fs_protocol)
    dataset_name = Path(root).name

    n5_paths = tuple(filter(fs.isdir, fs.glob(root +  '/*n5*/*/*')))
    precomputed_paths = tuple(filter(fs.isdir, fs.glob(root +  '/neuroglancer/em/*')))
    mesh_paths = tuple(filter(fs.isdir, fs.glob(root +  '/neuroglancer/mesh/*')))

    output_mesh_sources = [MeshSource(str(Path(mp).relative_to(root)), Path(mp).stem, dataset_name, 'neuroglancer_legacy_mesh') for mp in mesh_paths]

    volume_paths = (*n5_paths, *precomputed_paths)
    volume_path_stems = tuple(Path(s).stem for s in volume_paths)
    
    query = {"datasetName": dataset_name}
    db_volume_sources: List[VolumeSource] = typed_list_from_mongodb(mongo_addr, db, VolumeSource, query)
    if len(db_volume_sources) < 1:
        raise ValueError(f"No sources found in the database using query {query}")
    
    db_source_dict = {s.name: s for s in db_volume_sources}
    output_volume_sources: List[VolumeSource] = []
    for sk, sv in db_source_dict.items():
        if sk not in volume_path_stems:
            print(f'Warning: could not find an extant volume on the filesystem matching this VolumeSource from the database: {sv}. This volume will not be added to the dataset index.')
        else:
            input_source: VolumeSource = replace(sv)
            vol_path = volume_paths[volume_path_stems.index(sk)]
            input_source.path = str(Path(vol_path).relative_to(root))
            input_source.containerType = infer_container_type(vol_path)
            output_volume_sources.append(input_source)

    db_views: List[DatasetView] = []
    db_views = typed_list_from_mongodb(mongo_addr, db, DatasetView, query)
    
    accepted_views = []
    for v in db_views: 
        missing = set(v.volumeKeys) - set(db_source_dict.keys())
        if len(missing) > 0:
            print(f'This view contains volumes: {missing} that could not be found in the volume source database and thus will not be included in the index: {v}')
        else:
            accepted_views.append(v)

    index = DatasetIndex(name=dataset_name, volumes=output_volume_sources, meshes=output_mesh_sources, views=accepted_views)

    return index