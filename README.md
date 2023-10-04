# Controllability-Constrained-Latent-Models

## Setup

1. clone the repo:     
```
git clone https://github.com/suruchi1997/ControlledVAE.git
```
2. setup the anaconda environment:       
cuda: `conda env create --file env_cuda.yml`       
cpu: `conda env create --file env_cpu.yml`

## Reproducing Results

Regenerating model results for each experiment is divided into a few steps:

1. Training M baseline models. Since the baseline model may be sensitive to weight initialization, we provide a 
 --rs option to set the random seed. A few good seeds are available in `rseeds.txt`.
2. Training N models within a range of beta values based on each baseline model. Beta values 
have to be consistent across all seeds. We provide a range of beta values in `betas.txt`.
3. Evaluating all trained models.
4. Making plots.

## Experiments 

Both Pendulum and Cartpole follow the similar steps for training and evaluating the models:

1. `train.py` The base models and controllability-constrained models are trained and saved in the models/ folder.    
* (option A) Slower, in sequence training:   
   use `python train.py` to train all models in sequence. Will use `rseeds.txt` to train M base models,
and `betas.txt` to train N models for each rseed.    
* (option B) Parallelized training:
  1. Train M base models + beta=0 in parallel ```python train.py --rs {integer} --beta 0.0```   
    Slurm script for training all base models: ```sbatch train_parallel_base.sh```
  2. Train N models for each remaining beta value, using ```python train.py --rs {integer} --beta {float}```    
    Slurm script for training the rest of the models: ```sbatch train_parallel_ctrl.sh```
* (option C) Run B.ii in parallel for all rseed-beta combinations if the overhead of re-learning the base model is not a concern.
2. Then `python eval.py` is used to execute the evaluation process. Same options as for training.   
    slurm script for evaluation: ```sbatch eval_parallel.sh```    


## Making plots
1. `cd <experiment>`
2. `python ../csv_com.py` - generate csv files based on evaluation results. 
3. `python ../plots.py` - generate the "Control Cost Change in %" vs "Degree of Controllability" plot.

## Link to Datasets
https://drive.google.com/drive/u/0/folders/1piegH2gSNOG6WZtlOgSI3MLtDAUxj7ng
**Above is the link to datasets in order to regenerate the following results.The model can be data set oriented and has a lot of scope for improvement 
    
