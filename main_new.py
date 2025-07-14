import yaml
import argparse
import numpy as np
import metrics
from simulation_generator import simulation

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--init', help = 'Only generate the mesh (default: 0)', default = 0, type = int)
parser.add_argument('-g', '--gradient', help = 'Compute the term of the RANS equations as a post-processing (default: 0)', default = 0, type = int)
parser.add_argument('-v', '--vtk', help = 'Generate the VTK files from the simulation (default: 1)', default = 1, type = int)
parser.add_argument('-f', '--figure', help = 'Save an image of the airfoil in the simulation folder (default: 1)', default = 1, type = int)
parser.add_argument('-p', '--params', help = 'Name of parameter file. (default: params.yaml)', default = "params.yaml", type = str)
args = parser.parse_args()

param_files = ["params.yaml", "params_high_vel.yaml", "params_high_vel_AoA.yaml"]
# global_paths = ["/local00/bioinf/airfrans/Simulations/", "/local00/bioinf/airfrans/Simulations_downscaled/"]
global_paths = ["/local00/bioinf/airfrans/Simulations_downscaled/"]

for glob_path in global_paths:
    for param_file in param_files:
        # if param_file == "params.yaml" and glob_path == "Simulations/":
        #     continue
        with open(param_file, 'r') as f: # hyperparameters of the model
            params = yaml.safe_load(f)

        if glob_path == "/local00/bioinf/airfrans/Simulations_downscaled/":
            params["y_h"] = 1.2e-3
            params["y_hd"] = 4e-2


        params['Uinf'] = np.round(params['u_in'], 3)
        params['aoa'] = np.round(params['aoa'], 3) # Angle of attack. (< 15)
        params['digits'] = tuple(np.round(params['digits'], 3)) # 4 or 5-digits for the naca airfoil.

        init_path = glob_path + 'airFoil2DInit/' # OpenFoam initial case path (don't forget the '/' at the end of the path)

        digits = ''
        for digit in params['digits']:
            digits = digits + '_' + str(digit) 
        path = glob_path + 'airFoil2D_' + params['turbulence'] + '_' + str(params['Uinf']) + '_' + str(params['aoa']) + digits + '/' 

        simulation(init_path, path, params, just_init = bool(args.init), figure = bool(args.figure), compute_grad = bool(args.gradient), VTK = bool(args.vtk))
        res = metrics.plot_residuals(path, params)
        coef = metrics.plot_coef_convergence(path, params)
