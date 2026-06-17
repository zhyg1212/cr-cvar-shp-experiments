# PJM real-data experiment

This folder contains a MATLAB pipeline for the PJM day-ahead / real-time LMP
case study.

## Data

The experiment expects two CSV files:

- `data_pjm/pjm_da_lmp_raw.csv`
- `data_pjm/pjm_rt_lmp_raw.csv`

They should contain PJM Data Miner 2 feeds:

- `da_hrl_lmps`: Day-Ahead Hourly LMP
- `rt_hrl_lmps`: Real-Time Hourly LMP

PJM Data Miner requires a subscription key. If you have one, set:

```matlab
setenv('PJM_API_KEY', 'your-key-here')
```

Then run:

```matlab
cfgNodes = {'WESTERN HUB', 'EASTERN HUB', 'AEP', 'DOMINION'};
fetch_pjm_lmp_data('data_pjm', datetime(2025,5,1), datetime(2025,8,1), ...
    cfgNodes, getenv('PJM_API_KEY'));
```

Alternatively, export the two feeds manually from PJM Data Miner and save them
with the expected file names.

## Experiment

Run:

```matlab
run_pjm_realdata_experiment
```

The script uses a rolling-window design. For each test day in July 2025, the
previous 60 days of real-time LMPs estimate the training distribution. The
test-day day-ahead LMPs define first-stage prices, and the test-day real-time
LMPs define the realized second-stage values.

The output files are:

- `results/pjm_realdata_results.csv`
- `results/pjm_realdata_summary.csv`
