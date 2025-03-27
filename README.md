# Incremental-Variable-Construction-based-Cross-Mapping
This is the code for Predicting Direct Causal Links in Complex Systems.

## Software Requirements

### OS Requirements

The package has been tested on the following systems:

    Windows
    Linux

### Python Dependencies
- **Python Version**: Python 3.11.7
- All required libraries are listed in `requirements.txt`

      numpy==1.24.3
      dcor==0.6
      scipy==1.11.1
      statsmodels==0.14.0
      scikit-learn==1.3.0

## Installation Guide
Codes can be directly used while all required libraries are installed.

## Project Structure

The project is organized as follows:

- **`README.md`**: Project documentation and usage guide.

- **`requirements.txt`**: All required Python packages.

- **`InVaXMap.py`**: Main functions of InVaXMap method.

- **`mainf_biosphere_atmosphere_systems.py`**: Codes for experiments within the biosphere-atmospheric systems.

- **`mainf_brain_systems.py`**: Codes for experiments within the brain systems.

- **`mainf_dream4.py`**: Codes for experiments within the DREAM4 datasets.

- **`mainf_mdLV.py`**: Codes for experiments within the multi-species discrete Lotka-Volterra competition systems.

- **`Biosphere-atmosphere systems/`**: Data files for biosphere-atmosphere systems.

- **`Brain systems/`**: Data files for brain systems.

- **`Gene regulatory systems/`**: Data files for DREAM4 datasets.

- **`Multi-species discrete Lotka-Volterra competition model/`**:
  - `data_generation.m`: MATLAB code to generate multi-species discrete Lotka-Volterra competition systems for three modes: cycle mode, random mode, and structural stochastic mode.

## License
This project is licensed under the MIT License.
