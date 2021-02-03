from dataclasses import  dataclass
from typing import Any, Callable, Dict, Tuple, Optional, List, TypeVar, Sequence
from fibsem_tools.io.io import access_precomputed
import numpy as np
import os
from xarray_multiscale.multiscale import multiscale, get_downscale_depth
from xarray_multiscale.metadata import cosem_ome, neuroglancer
from xarray_multiscale.reducers import mode
import tensorstore as ts
from xarray import DataArray
from distributed import Client
from cosem_flows.attrs import (
    VolumeSource,
    VolumeIngest,
    MultiscaleSpec,
    VolumeStorageSpec,
)
import distributed
import time
from tqdm import tqdm
from dask_janelia import get_cluster
from fibsem_tools.io import populate_group
import click
from pymongo import MongoClient
import dask.array as da
import toolz as tz

def downscale_slices(slices: Tuple[slice], scale_factors: Sequence[int], trim_stop: bool=False) -> Tuple[slice]:
    results = []
    for slic, factor in zip(slices, scale_factors):
        if (trim_stop == False) and (slic.start % factor != 0 or slic.stop % factor != 0):
            raise ValueError(f'Slice bounds {slic.start, slic.stop} are not evenly divisible by {factor}')
        results.append(slice(slic.start // factor, slic.stop // factor, slic.step))
    return tuple(results)

def check_initialized_chunks(path: str) -> bool:
    from fibsem_tools.io import read
    r = read(path)
    # i check whether nchunks is less than or equal to initialized chunks because it seems that the zarr python library
    # is a little generous with what it considers a chunk, and thus it's possible to have more chunks initialized than total chunks,
    # e.g. if some .nfsblablabla files are floating around from other processes
    return all(i.nchunks <= i.initialized for n,i in r.items())

def sequential_rechunk(source: Any, target: Any, slab_size: Tuple[int,...], intermediate_chunks: Tuple[int,...], client: distributed.Client, num_workers: int, streaming=False) -> List[None]:
    """
    Load slabs of an array into local memory, then create a dask array and rechunk that dask array, then store into 
    chunked array storage.
    """
    results = []
    futures = []
    slices = da.core.slices_from_chunks(source.rechunk(slab_size).chunks)
    client.cluster.scale(num_workers)
    for sl in slices:
        arr_in = source[sl].compute(scheduler='threads')
        darr_in = da.from_array(arr_in, chunks=intermediate_chunks)
        store_op = da.store(darr_in, target, regions=sl, compute=False, lock=None)
        if streaming:
            futures.append(client.compute(store_op))
        else:
            results.extend(client.compute(store_op).result())
    if streaming:
        results = client.gather(futures)
    client.cluster.scale(0)
    return results

def sequential_multiscale(source: Any, targets: Any, reducer: Any, scale_factors: Tuple[int,...], multiscale_levels: List[int], slab_size: Tuple[int,...], intermediate_chunks: Tuple[int,...], client: distributed.Client, num_workers: int, streaming=False) -> List[None]:
    """
    Load slabs of an array into local memory, then create a dask array and rechunk that dask array, then store into 
    chunked array storage.
    """
    results = []
    futures = []
    slices = da.core.slices_from_chunks(source.rechunk(slab_size).chunks)
    client.cluster.scale(num_workers)
    
    for idx, sl in tqdm(tuple(enumerate(slices))):
        print(f'Reading in data subrange with shape {source[sl].shape}')
        arr_in = source[sl].compute(scheduler='threads')
        darr_in = DataArray(da.from_array(arr_in, chunks=intermediate_chunks))
        valid_depth = get_downscale_depth(darr_in.shape, scale_factors)    
        levels = np.array(multiscale_levels)[np.array(multiscale_levels) < valid_depth]
        if valid_depth == 0:
            multi = [darr_in]
        else:
            multi = tz.get(levels, multiscale(darr_in, reducer, scale_factors))
        
        multi_rechunked = [m.chunk(intermediate_chunks) for m in multi]
        scales = [np.array(scale_factors) ** idx for idx in range(len(multi))]
        regions = [downscale_slices(sl, scale_factors=scale, trim_stop=True) for scale in scales]
        print(f'saving to storage with regions {regions}')
        store_op = da.store([m.data for m in multi_rechunked], targets[:len(multi)], regions=regions[:len(multi)], compute=False, lock=None)
        if streaming:
            futures.append(client.compute(store_op))
        else:
            results.extend(client.compute(store_op).result())
    if streaming:
        results = client.gather(futures)
    client.cluster.scale(0)
    return results


def populate_precomputed(store_path: str, multi: Sequence[Any], level_names: Sequence[str], jpeg_quality: int, chunks: Tuple[int, ...], channel: int=0) -> List[Any]:
    store_arrays = []
    for idx,scale in enumerate(multi):
            if idx == 0: 
                mode = 'w' 
            else: 
                mode = 'a'
            resolution = [abs(float(scale.coords[d][1].values - scale.coords[d][0].values)) for d in scale.dims]
            arr_precomputed = access_precomputed(store_path, 
            key=level_names[idx], 
            scale_index = idx,
            mode=mode, 
            array_type='image', 
            dtype=scale.dtype.name, 
            num_channels=1, 
            shape=scale.shape, 
            resolution=resolution,
            encoding='jpeg', 
            jpeg_quality=jpeg_quality,
            chunks=chunks)
            arr_precomputed = arr_precomputed[ts.d["channel"][channel]]
            store_arrays.append(arr_precomputed)
    return store_arrays
    

def flip_axis(arr: DataArray, axis: str) -> DataArray:
    arr2 = arr.copy()
    idx = arr2.dims.index(axis)
    arr2.data = np.flip(arr2.data, idx)
    return arr2


reducers: Dict[str, Callable[[Any], Any]] = {"mean": np.mean, "mode": mode}
mutations: Dict[str, Callable[[Any], Any]] = {"flip_y": lambda v: flip_axis(v, "y")}
ndim = 3
# debug 
#debug_crop = {'z' : slice(2048), 'y': slice(-1), 'x': slice(-1)}
debug_crop = None

access_modes = ('write', 'metadata')
scale_factors = (2,) * ndim
read_chunks = (512,) * 3
input_chunks = (64, -1, -1) 
output_chunks = (64,) * ndim


# mongodb variables
un = "root"
pw = "root"
addr = "cosem.int.janelia.org"
db = "sources"
mongo_addr = f"mongodb://{un}:{pw}@{addr}"

# this stuff is due to get promoted to module
import dask.array as da
import zarr 
import numpy as np
import dask

def store_block(source, target, block_info=None):
    chunk_origin = block_info[0]['array-location']
    slices = tuple(slice(start, stop) for start, stop in chunk_origin) 
    target[slices] = source
    return np.expand_dims(0, tuple(range(source.ndim)))


def store_blocks(sources, targets):
    result = []
    if isinstance(sources, dask.array.core.Array):
        sources = [sources]
        targets = [targets]
    for source, target in zip(sources, targets):
        out_chunks = tuple((1,) * len(c) for c in source.chunks)
        result.append(da.map_blocks(store_block, source, target, chunks=out_chunks, dtype='int64'))
    return result  


@dataclass
class MultiscaleSavePlan:
    mode: str
    input_chunks: Tuple[int, ...]
    output_chunks: Tuple[int, ...]
    jpegQuality: int = 90

T = TypeVar('T')

def typed_list_from_mongodb(mongo_addr: str, db: str, cls: T, query: Dict[str, str]) -> List[T]:
    with MongoClient(mongo_addr) as client:
        retrieved = client[db][cls.__name__].find(query)
    results = []
    for r in retrieved:
        r.pop('_id')
        results.append(cls.fromDict(r))
    return results


def makeVolumeIngest(source: VolumeSource, output_path: str) -> VolumeIngest:
    mutation: Optional[str] = None
    reduction = "mean"
    dataPath = ""
    containerPath = ""

    if source.dataType == "uint8" and source.contentType == "em":
        containerType = "precomputed"
        dataPath = ""
        containerPath = os.path.join(output_path, source.datasetName, "neuroglancer", "em", f"{source.name}.precomputed")
    else:
        containerType = "n5"
        containerPath = os.path.join(output_path, source.datasetName, f"{source.datasetName}.n5")
        if source.contentType == "em":
            content_dir = "em"
        elif source.contentType == "lm":
            content_dir = "lm"
        elif source.contentType in {"prediction", "gt", "segmentation", "analysis"}:
            content_dir = "labels"
        else:
            raise ValueError(
                f"content type {source.contentType} cannot be dispatched to a path"
            )
        dataPath = os.path.join(content_dir, source.name)
    storageSpec = VolumeStorageSpec("file", containerType, containerPath, dataPath)

    if source.contentType != "em" and source.datasetName in ("jrc_hela-1", "jrc_hela-2", 'jrc_hela-3', 'jrc_jurkat-1', 'jrc_macrophage-2'):
        mutation = "flip_y"

    if source.contentType == "segmentation":
        reduction = "mode"
    if source.datasetName in ('jrc_cos7-11') and source.contentType == 'prediction':
        # these prediction volumes are 4nm grid spacing for 8nm data, so we don't need s0
        levels = list(range(1,7))
    else:
        levels = list(range(0,6))
    multiscaleSpec = MultiscaleSpec(reduction, levels, (2, 2, 2))

    ingest = VolumeIngest(source, multiscaleSpec, storageSpec, mutation)
    return ingest


def ingest_source(ingest: VolumeIngest, save_plan: MultiscaleSavePlan, num_workers: int, destination: str):
    source = ingest.source
    parallel_reads = source.containerType in ('n5', 'zarr', 'precomputed')
    print(f'Using mutation `{ingest.mutation}`')
    mutation = mutations.get(ingest.mutation, lambda v: v)
    
    darr: DataArray = mutation(source.toDataArray(chunks=read_chunks))
    if parallel_reads:
        # we can handle most isotropic-ish chunking, but dask has a bug for certain chunking schemes and 
        # this gets around it
        darr: DataArray = mutation(source.toDataArray()).chunk(read_chunks)

    if debug_crop:
        darr = darr[debug_crop]
    print(f'Source data: {darr.data}, found at {source.path}')
    

    input_chunks = save_plan.input_chunks
    reducer = reducers[ingest.multiscaleSpec.reduction]
    multiscale_levels = ingest.multiscaleSpec.levels
    scale_factors = ingest.multiscaleSpec.factors

    multi = multiscale(
        darr, reduction=reducer, scale_factors=scale_factors)
    multi = tz.get(multiscale_levels, multi)
    level_names = [f"s{level}" for level in multiscale_levels]
    container_path = os.path.join(destination, ingest.storageSpec.containerPath) 

    if ingest.storageSpec.containerType == "n5":
        group_path = ingest.storageSpec.dataPath
        print(f"Preparing the store {os.path.join(container_path, group_path)}")
        
        group_attrs = {**cosem_ome.GroupMeta.fromDataArraySequence(multi, paths=level_names).asdict(), **neuroglancer.GroupMeta.fromDataArraySequence(multi).asdict()}
        array_attrs = [cosem_ome.ArrayMeta.fromDataArray(m).asdict() for m in  multi]
        store_group, store_arrays = populate_group(
            container_path,
            group_path,
            multi,
            array_paths=level_names,
            chunks=(save_plan.output_chunks,) * len(multi),
            group_attrs = group_attrs,
            array_attrs=array_attrs,
        )
    elif ingest.storageSpec.containerType == "precomputed":
        group_path = ''
        print(f"Preparing the store {container_path}")
        if darr.dtype.name not in ("uint8"):
            raise ValueError("Only uint8 supported at this time")
        store_group = None
        # create the precomputed arrays with transposed data
        store_arrays = populate_precomputed(container_path, [m.T for m in multi], level_names, save_plan.jpegQuality, save_plan.output_chunks)
        # transpose to bring the precomputed arrays into z,y,x order
        store_arrays = [arr.T for arr in store_arrays]
    else:
        raise ValueError(f'Could not create array storage within the container type {ingest.storageSpec.containerType}')
    
    if save_plan.mode == 'data':
        with get_cluster() as clust, Client(clust) as cl:
            print(cl.cluster.dashboard_link)            
            start = time.perf_counter()
            if parallel_reads:
                cl.cluster.scale(num_workers)
                result = cl.compute(store_blocks([m.chunk(input_chunks).data for m in multi], store_arrays), sync=True)
                #result = cl.compute(da.store([m.chunk(input_chunks).data for m in multi], store_arrays, compute=False, lock=None)).result()
            else:
                result = sequential_multiscale(darr.data, store_arrays, reducer, scale_factors, multiscale_levels, slab_size=input_chunks, intermediate_chunks=(256,) * 3, client=cl, num_workers=num_workers, streaming=False)            
            print(f'Completed in {time.perf_counter() - start}s')
            
        return result

@click.command()
@click.option("-q", "--query", required=True, type=str)
@click.option("-d", "--destination", required=True, type=str)
@click.option("-w", "--workers", required=False, type=int, default=64)
@click.option("-m","--mode", type=str, default='data')
def ingest_source_cli(query: str, destination: str, mode: str, workers: int):
    query_result = typed_list_from_mongodb(mongo_addr, db, VolumeSource, eval(query))
    if len(query_result) == 0:
        raise ValueError(f'Could not find a single dataset using the query {query}')
    sources = query_result
    for source in sources:
        ingest = makeVolumeIngest(source, destination)
        plan = MultiscaleSavePlan(mode=mode, input_chunks=input_chunks, output_chunks=output_chunks)
        pth = os.path.join(ingest.storageSpec.containerPath, ingest.storageSpec.dataPath)
        print(f'Ingesting store at {pth}...')
        ingest_source(ingest, plan, num_workers=workers, destination=destination)
        
if __name__ == '__main__':
    ingest_source_cli()