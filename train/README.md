# Train
this is the train code implementation of “Seismic Data Reconstruction Based On Conditional Constraint Diffusion Model”

#Installation
run:
```
pip install -e .
```
# Training models

## Preparing Data

The training code reads npy data from a directory of data files. 

For creating your own dataset, simply dump all of your datas into a directory with ".npy" extensions. 

Simply pass --data_dir path/to/datas to the training script, and it will take care of the rest.

## training
To train your model, you should first decide some hyperparameters. We will split up our hyperparameters into three groups: model architecture, diffusion process, and training flags. Here are some reasonable defaults for a baseline:
```
MODEL_FLAGS="--data_size 64 --num_channels 128 --num_res_blocks 3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128"
```
Once you have setup your hyper-parameters, you can run an experiment like so:
```
python scripts/image_train.py --data_dir path/to/images $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```
You may also want to train in a distributed manner. In this case, run the same command with mpiexec:
```
mpiexec -n $NUM_GPUS python scripts/image_train.py --data_dir path/to/images $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```
When training in a distributed manner, you must manually divide the --batch_size argument by the number of ranks. In lieu of distributed training, you may use --microbatch 16 (or --microbatch 1 in extreme memory-limited cases) to reduce memory usage.

The logs and saved models will be written to a logging directory determined by the OPENAI_LOGDIR environment variable. If it is not set, then a temporary directory will be created in /tmp.