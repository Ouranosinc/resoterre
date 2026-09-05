import yaml
from pathlib import Path
import yaml

from resoterre import PROJECT_ROOT
from resoterre.hybrid_data_loaders.crcm_emulator_data_loader import CRCMEmulatorDataset

config_path = PROJECT_ROOT / "configs" / "crcm_emulator" / "crcm_emulator.yaml"
print(config_path)
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

print(config.keys())
path_data = Path(config["path_data"])
path_preprocessed = path_data / config["path_preprocessed"]
path_output = path_data / config["path_output"]

assert path_preprocessed.exists(), path_preprocessed

print(path_data)
print(path_preprocessed)
print(path_output)

dataset = CRCMEmulatorDataset(
    path_gcm_preprocessing = path_preprocessed, 
    path_crcm_preprocessing = path_preprocessed, 
    simulations = [config["preprocessing_simulations"][0], config["preprocessing_simulations"][1]], 
    gcm_variables = config["gcm_training_variables"], 
    crcm_variables = config['crcm_training_variables'], 
    time_periods = config["training_periods"]
)

sample = dataset[0]
print(sample)


# data: input (2, 76, 76), target (1, 608, 608), year, month, day, **emission_data (CO2, CH4, N2O, CFC12, CFC11_eq)

# Q1: What are the stats for missing data in the inputs?
# Step 1: Check for NaN values in the input data
# Step 2: Stats
# - Percentage of missing values in the input data for 1000 vs 850 hpa
# - How large are the average clusters? 
# - 

# ==========

# Q2: Mismatch between 

# Step 1: Upscale target data to match input resolution
# Step 2: Compare the upscaled target data with the input data
# - MSE
# - MAE
# - Correlation coefficient
# - Visual inspection (plot a few examples)
# - Histogram of differences between upscaled target and input data


# - mean stdv of on inputs / targets 
# - min, max, range, 

# - 1971 - 2000 range 