from copy import deepcopy
import yaml
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import metrics
from simulation_generator import simulation


def job(glob_path, params, sim):
    params['Uinf'] = np.round(float(sim[0]), 3) 
    params['aoa'] = np.round(float(sim[1]), 3) # Angle of attack. (< 15)
    # params['digits'] = tuple(np.round(map(float, sim[2:]), 3)) # 4 or 5-digits for the naca airfoil.
    params['digits'] = tuple(np.round(list(map(float, sim[2:])), 3))

    if np.abs(params['aoa']) <= 10:
        params['n_iter'] = 20000
    else:
        params['n_iter'] = 40000

    init_path = glob_path + 'airFoil2DInit/' # OpenFoam initial case path (don't forget the '/' at the end of the path)

    digits = ''
    for digit in params['digits']:
        digits = digits + '_' + str(digit) 
    path = glob_path + 'airFoil2D_' + params['turbulence'] + '_' + str(params['Uinf']) + '_' + str(params['aoa']) + digits + '/' 
    
    simulation(init_path, path, params, figure = True, compute_grad = True, VTK = True)
    res = metrics.plot_residuals(path, params)
    coef = metrics.plot_coef_convergence(path, params)
    return path


if __name__ == "__main__":
    # Global path where the airFoil2DInit folder is and where the simulations are gonna be done.
    glob_path = '/local00/bioinf/airfrans/full_dataset_downscaled/'

    # Num workers (each worker uses 16 MPI processes, so on jackal n_workers should be < 12)
    n_workers = 10

    with open('params_coarse.yaml', 'r') as f: # hyperparameters of the model
        params = yaml.safe_load(f)

    # Load simulations
    with open('./manifest.json', 'r') as f: # hyperparameters of the model
        dataset_sims = json.load(f)

    all_sims = dataset_sims["full_train"] + dataset_sims["full_test"]

    print(len(all_sims))

    sim_configs = []
    for sim_folder in all_sims:
        if ".json" in sim_folder:
            continue
        config = sim_folder.split("_")[2:]  # remove first two (airfoil2D, SST)
        sim_configs.append(config)

    with ProcessPoolExecutor(max_workers=n_workers) as exec:
        futures = [exec.submit(job, glob_path, deepcopy(params), sim_config) for sim_config in sim_configs]
        for f in tqdm(as_completed(futures), desc="Running simulations", total=len(sim_configs)):
            case = f.result()
            print(f"Finished {case}")
