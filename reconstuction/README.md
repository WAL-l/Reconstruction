# Reconstuct
this is the reconstruction code implementation of “Seismic Data Reconstruction Based On Conditional Constraint Diffusion Model”

## Setup

### 1. Code

```bash
git clone https://github.com/WAL-l/seismic_reconstruction.git
```

### 2. Environment
```bash
pip install numpy torch blobfile tqdm pyYaml pillow    # e.g. torch 1.7.1+cu110.
```

### 4. Run example
```bash
python test.py --conf_path confs/example.yml
```
Find the output in `./log/example/inpainted`

*Note: After refactoring the code, we did not reevaluate all experiments.*

<br>


## Details on data



**How to prepare the test data?**

one file for the data which is ".npy"
one file for mask
The masks have the value 255 for known regions and 0 for unknown areas (the ones that get generated).



