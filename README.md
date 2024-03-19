# Reconstruction
this is the code implementation of “[Seismic Data Reconstruction Based On Conditional Constraint Diffusion Model](https://ieeexplore.ieee.org/document/10453526)”

# Result

### Synthetic data

[//]: # (![Synthetic data]&#40;./docs/Synthetic.png&#41;)
<img width="1000" alt="Synthetic data" src="./docs/Synthetic.png">

### Field data

<img width="1000" alt="Synthetic data" src="./docs/Field.png">

# reconstruction your data

1. Use "train" to train your model and get the ".pt" model file.
2. Place the model file under "data/mod" in "reconstruction" and configure "confs" with the required configuration information.
3. Use "reconstruction" to reconstruct your data.

# Cite us

### if this work is helpful for you, please cite
```
@ARTICLE{10453526,
  author={Deng, Fei and Wang, Shuang and Wang, Xuben and Fang, Peng},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={Seismic Data Reconstruction Based on Conditional Constraint Diffusion Model}, 
  year={2024},
  volume={21},
  number={},
  pages={1-5},
  keywords={Mathematical models;Data models;Interpolation;Encoding;Vectors;Training;Predictive models;Diffusion model;neural network;seismic data reconstruction},
  doi={10.1109/LGRS.2024.3371675}}
```
