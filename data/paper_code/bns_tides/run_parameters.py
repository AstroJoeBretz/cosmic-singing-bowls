import os, yaml
import numpy as np
from string import Template


def default_parameters(
    param_type,
    is_adiabatic=False
):
    """
    """
    params_dict = {}
    
    params_dict['baseline'] = {
        'M_cano': 1.33, # Solar Mass
        'R1': 11.7e5, 'R2': 11.7e5, # Centimeters
        'Mchirp': 1.157, # Solar Mass
        'q': 0.942, # Dimensionless
        't_coal': 0., # Milliseconds
        'phi_coal': 0., # Radians
        'lam_cano': 400., # Dimensionless
        'wab_cano': 0.0786, # Dimensionless
        'Anl_cano': 0., # Dimensionless
        'wnl_ref': np.pi*1000, # pi Hz (GW)
        'Omega_S1': 0., # Hz
        'Omega_S2': 0., # Hz
        'chi1': 0., # Dimensionless z component of spin
        'chi2': 0., # Dimensionless z component of spin
        'gr_flag': 0, # {0: no GR, 1: GR}
    }

    params = params_dict[param_type]

    if is_adiabatic:
        params['wab_cano'] = 100

    param_types = list(params_dict.keys())
    assert param_type in param_types, f"Specify valid parameters type: {param_types}"
    
    return params


def save_parameters_to_yaml(
    model_parameters: dict, 
    effect_parameters: dict, 
    mcmc_parameters: dict, 
    filename: str
):
    """
    Saves a YAML file and returns the saved filename.

    Args:
        model_parameters (dict): Dict of all model parameters.
        effect_parameters (dict): Dict of all model parameters with effect.
        mcmc_parameters (dict): Dict of mcmc parameters.
        filename (str): Name of output saved YAML file.

    Prints:
        Name of saved YAML file.
    """
    model_params = model_parameters.copy()
    effect_params = effect_parameters.copy()
    mcmc_params = mcmc_parameters.copy()

    # Shorten wnl_ref to precision of 4
    model_params['wnl_ref'] = round(model_params['wnl_ref'], 4)
    effect_params['wnl_ref'] = round(effect_params['wnl_ref'], 4)

    output = {
        'model_parameters': model_params,
        'effect_parameters': effect_params,
        'mcmc_parameters': mcmc_parameters
    }

    # Extract path from filename
    path, file = os.path.split(filename)

    if path:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Directory '{path}' created.")
    
    with open(filename, 'w') as f:
        yaml.dump(output, f, sort_keys=False)
    
    print(f'Parameters saved to {filename}')


def load_parameters_from_yaml(file_path):
    """
    Loads a YAML file and returns the data as a Python object.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        dict or list: The data from the YAML file, or None if an error occurs.
    """
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            return data
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None


def save_job_sbatch(
    job_parameters: dict
):
    """
    """
    sbatch_template = Template("""#!/bin/bash
##
## Lines starting with #SBATCH are read by Slurm. Lines starting with ## are comments.
## All other lines are read by the shell.

## Specify the account to use; you must use your group account.
#SBATCH --account=$group_account

## Use the test partition.
#SBATCH --partition=$partition

## Number of CPUs to allocate. Note that in the slurm, CPUs correspond to
## logical processors, not physical cores.
#SBATCH --cpus-per-task=$cpus

#SBATCH --mem=$mem

## Maximum job run time. Format is days-HH:MM:SS.
#SBATCH --time=$time

## Set job name.
#SBATCH --job-name=$job_name

## Set output log and error log filenames. 
## %j is the job number and %x is the job name set above 
#SBATCH --output=$output_path/%x-%j.out
#SBATCH --error=$output_path/%x-%j.err

## Remove one # and fill out the following lines if you want to receive emails.
##SBATCH --mail-user=      # enter your email to recieve email notifications
##SBATCH --mail-type=ALL                 # specify conditions on which to recieve emails

## Run 'man sbatch' for more information on the options above.
## Below, enter commands needed to execute your workload

# Print out the date
date
# Print which partition we are using
echo "partition = ${SLURM_JOB_PARTITION}"
# Print the compute node's name
echo "Hello from $(hostname -s)"

## module load Mamba/23.11.0-0
source ~/.bashrc
mamba activate hmc-env
## module load texlive/20230313-GCC-13.2.0
module load tools/texlive/tl-2021

python $python_file $filename

date
""")
    output_path, _ = os.path.split(job_parameters['filename'])
    sbatch_path = os.path.join(output_path, 'job.sbatch')
    job_parameters['output_path'] = output_path
    job_sbatch = sbatch_template.safe_substitute(job_parameters)
    # Write sbatch file
    with open(sbatch_path, 'w') as file:
        file.write(job_sbatch)
    
    print(f"job.sbatch saved to {job_parameters['filename']}")


def classify_vector_type(vector):
    """
    Classifies a vector as 'linear' or 'logarithmic' based on the ratio of consecutive elements.

    Args:
        vector (list or numpy.ndarray): A list or numpy array of numerical values.

    Returns:
        str: 'linear' if the vector is classified as linear, 'logarithmic' if classified as logarithmic,
             or 'unknown' if classification is not possible.
    """
    vector = np.array(vector)
    if np.any(vector <= 0):
      return "unknown"
    ratios = vector[1:] / vector[:-1]
    if np.all(np.isclose(ratios, ratios.mean())):
        return "logarithmic"
    diffs = np.diff(vector)
    if np.all(np.isclose(diffs, diffs.mean())):
        return "linear"
    return "unknown"


def extract_parameters_from_args(model_args):
    """
    """
    model_params = model_args.copy()

    freq = mcmc_params['freq']
    freq_details = {
        'min': round(float(np.min(freq)), 4),
        'max': round(float(np.max(freq)), 4),
        'length': len(freq),
        'spacing': classify_vector_type(freq),
        'avg_diff': round(float(np.mean(np.diff(freq))), 4),
        'mdn_diff': round(float(np.median(np.diff(freq))), 4)
    }
    
    model_params.pop('freq')
    model_params['freq'] = freq_details

    M1, M2 = model_params['M1']/Ms, model_params['M2']/Ms
    # model_params['q'] = round(M1/M2, 4)
    # model_params['Mchirp'] = round((M1**3 * M2**3 / (M1+M2))**(1/5), 4)
    
    model_params['M_c'] = round(model_params['M_c']/Ms, 4)
    model_params['M1'] = round(M1, 4)
    model_params['M2'] = round(M2, 4)

    model_params['wnl_ref'] = round(model_params['wnl_ref'], 4)

    return model_params

