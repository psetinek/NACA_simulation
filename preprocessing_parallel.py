import os
import os.path as osp
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor, as_completed
import pyvista as pv
from tqdm import tqdm


# parameters
# N_WORKERS = 64
N_WORKERS = 64
SIMULATIONS_FOLDER = "/local00/bioinf/airfrans/full_dataset_downscaled/"
OUTPUT_FOLDER = "/local00/bioinf/airfrans/full_dataset_preprocessed/"

CLIP_BOUNDS = (-2, 4, -1.5, 1.5, 0, 1)
SLICE_NORMAL = (0, 0, 1)
SLICE_ORIGIN = (0, 0, 0.5)

INTERNAL_FIELDS = ["p", "U", "nut", "implicit_distance"]
AERO_FIELDS = ["p", "U", "nut", "Normals"]
FREE_FIELDS = ["p", "U", "nut"]


def select_fields(ds, names):
    """Return a copy of `ds` carrying *only* the arrays in `names`."""
    out = ds.copy()
    # point‐data
    for arr in list(out.point_data.keys()):
        if arr not in names:
            out.point_data.remove(arr)
    # cell‐data
    for arr in list(out.cell_data.keys()):
        if arr not in names:
            out.cell_data.remove(arr)
    return out


def preprocess_simulation(sim_name, progress_bar=False):
    sim_folder = osp.join(SIMULATIONS_FOLDER, sim_name)
    t_end = str([d for d in (Path(sim_folder) / "VTK").iterdir() if d.is_dir()][0]).split("_")[-1]
    internal_path = osp.join(sim_folder, "VTK", sim_name + "_" + t_end, "internal.vtu")
    aerofoil_path = osp.join(sim_folder, "VTK", sim_name + "_" + t_end, "boundary", "aerofoil.vtp")
    freestream_path = osp.join(sim_folder, "VTK", sim_name + "_" + t_end, "boundary", "freestream.vtp")

    # load raw files
    internal = pv.read(internal_path)
    aerofoil = pv.read(aerofoil_path)
    freestream = pv.read(freestream_path)

    # internal preprocessing
    internal = internal.clip_box(bounds=CLIP_BOUNDS, crinkle=True, progress_bar=progress_bar, invert=False)  # clip box
    internal = internal.slice(normal=SLICE_NORMAL,
                        origin=SLICE_ORIGIN,
                        generate_triangles=False,
                        progress_bar=progress_bar)  # slice to 2D
    internal = internal.compute_implicit_distance(aerofoil)  # compute sdf to airfoil
    internal = select_fields(internal, INTERNAL_FIELDS)  # only keep necessary fields

    # aerofoil preprocessing
    aerofoil = aerofoil.compute_normals(point_normals=True, cell_normals=True, flip_normals=False)  # normals
    aerofoil = aerofoil.slice(normal=SLICE_NORMAL,
                        origin=SLICE_ORIGIN,
                        generate_triangles=False,
                        progress_bar=progress_bar)  # slice to 2D
    aerofoil = select_fields(aerofoil, AERO_FIELDS)  # only keep necessary fields
    aerofoil = aerofoil.compute_cell_sizes(area=False, volume=False)  # compute length (arc length)

    # freestream processing
    freestream = freestream.slice(normal=SLICE_NORMAL,
                        origin=SLICE_ORIGIN,
                        generate_triangles=False,
                        progress_bar=progress_bar)  # slice to 2D
    freestream = select_fields(freestream, FREE_FIELDS)  # only keep necessary fields

    # save preprocessed data
    os.makedirs(osp.join(OUTPUT_FOLDER, sim_name), exist_ok=True)
    pv.save_meshio(osp.join(OUTPUT_FOLDER, sim_name, sim_name + "_internal.vtu"), internal)
    aerofoil.save(osp.join(OUTPUT_FOLDER, sim_name, sim_name + "_aerofoil.vtp"))
    freestream.save(osp.join(OUTPUT_FOLDER, sim_name, sim_name + "_freestream.vtp"))

    return sim_name


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=N_WORKERS) as exec:
        futures = []
        sim_names = list(os.listdir(SIMULATIONS_FOLDER))
        for sim_name in sim_names:
            if "Init" in sim_name:
                continue
            futures.append(exec.submit(preprocess_simulation, sim_name, False))
        for future in tqdm(as_completed(futures), desc="Postprocessing simulations", total=len(sim_names)):
            case = future.result()
            print(f"Finished {case}")
