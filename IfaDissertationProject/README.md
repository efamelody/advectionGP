# AdvectionGP Pollution Source Inference (Bushfire Case Study)

This project extends the [AdvectionGP](https://github.com/SheffieldML/advectionGP) framework to infer pollution sources during the 2019–20 bushfires in Australia using real MERRA-2 wind data and Aerosol Optical Depth (AOD) observations.

## 🔍 Overview

- Simulates backward particle tracing from satellite-like sensor regions.
- Incorporates real 3D wind data from NASA MERRA-2.
- Uses Gaussian Processes and adjoint-based inference to reconstruct source distributions.
- Includes support for both synthetic and real AOD data.

## 📁 Project Structure

```plaintext
advectionGP/
├── wind/                  # Wind model implementations
│   ├── realwind.py        # RealWind + RealWindNearestNeighbour classes
│   └── __init__.py
├── sensors/               # RemoteSensingModel for particle generation
├── models/                # MeshFreeAdjointAdvectionDiffusionModel, etc.
├── data/                  # Optional: contains sample NetCDF files or outputs
│   └── sample_wind.nc4
├── scripts/               # Custom driver scripts
│   └── run_inference.py
├── config.yaml            # External configuration file
└── README.md
```

## 📦 Requirements

Install dependencies (Python 3.8+ recommended):

```bash
pip install -r requirements.txt
```

Main packages used:
- `xarray`, `netCDF4` for reading NASA wind files
- `pyproj` for coordinate conversion
- `scipy`, `numpy`, `pandas`
- `matplotlib`, `cartopy` for plotting
- `pyyaml` if using `config.yaml`

## 📂 Wind Data Setup

This project uses NASA MERRA-2 data from the dataset:

> MERRA-2 tavg3_3d_asm_Nv: 3-Hourly, Model-Level, Assimilated Meteorological Fields  
> DOI: [10.5067/SUOQESM06LPK](https://doi.org/10.5067/SUOQESM06LPK)

### Option 1: Download Your Own

1. Go to [GES DISC](https://disc.gsfc.nasa.gov/datasets/M2T3NVASM_5.12.4/summary)
2. Download `.nc4` files for your date range
3. Place them in `data/wind/`

Update `config.yaml` with your data folder and file naming scheme.

### Option 2: Quick Test

Use the included `data/sample_wind.nc4` for a small-scale test.

## ⚙️ Configuration

Set parameters such as file paths, start date, vertical layers, and bounding box in:

```yaml
# config.yaml
data:
  wind_dir: "data/wind"
  file_prefix: "MERRA2_400.tavg3_3d_asm_Nv"
  file_ext: ".nc4"

wind_model:
  layer_range: [56, 68]
  bounding_box: [110, -45, 155, -10]
  start_date: "2019-10-01"
  num_days: 9
```

## 🚀 Running the Model



## 📊 Visualisation

Output maps and particle movements can be visualised using `matplotlib` and `cartopy` in your analysis notebook or plotting script.

## 📚 Citation

If using the MERRA-2 dataset:

> Global Modeling and Assimilation Office (GMAO) (2015),  
> MERRA-2 tavg3\_3d\_asm\_Nv: 3-Hourly, Model-Level, Assimilation,  
> NASA GES DISC, DOI: [10.5067/SUOQESM06LPK](https://doi.org/10.5067/SUOQESM06LPK)

## 🧠 Acknowledgements

- Based on [AdvectionGP by SheffieldML](https://github.com/SheffieldML/advectionGP)
- Wind data from NASA MERRA-2 via GES DISC
```


