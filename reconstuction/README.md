# Reconstuct
this is the reconstruction code implementation of “Seismic Data Reconstruction Based On Conditional Constraint Diffusion Model”

## Setup


### 2. Environment
```bash
pip install numpy blobfile tqdm pyYaml pillow 
```

### 4. Run example
```bash
python reconstruction.py --conf_path confs/reconstruction.yml
```
Find the output in `./log`


<br>


## Details on data



**How to prepare the test data?**

one file for the data which is ".npy"
one file for mask
The masks have the value 255 for known regions and 0 for unknown areas (the ones that get generated).



