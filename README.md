# Controllability-Constrained-Latent-Models

## Setup

1. clone the repo:     
```
git clone https://github.com/suruchi1997/Controllability-Constrained-Latent-Models.git
```
2. setup the anaconda environment:       
cuda: `conda env create --file env_cuda.yml`       
cpu: `conda env create --file env_cpu.yml`

## Training models

Regenerating model results for each experiment is divided into a few steps:
1. Training several identically initialized models on a range of beta values. 
Since the baseline model may be sensitive to weight initialization, we provide a 
 --rs option to set the random seed. A few good seeds are available in `rseeds.txt`. 
This step should be run several times with different seeds.

2. Evaluating all trained models.

### Pendulum
1. `pen_conv.py` The base models and controllability-constrained models are trained and saved in the conv_mul/ folder.
   ```
    python pen_conv.py --rs {integer}
   ```  
2. Then mpc_eval2.py is used to execute the evaluation process.

   ```
    python mpc_eval2.py
   ```
### CartPole
1. `cp_cont4.py` The  base models and controllability-constrained models are trained and saved in the conv_new/ folder.
   
   ```
    python cp_cont4.py --rs {integer}
   ```  
2. cp_mpc_eval1.py is used for evaluating the performance of the models.

   ```
    python cp_mpc_eval1.py
   ```

## Making plots
* csv_com.py can be used to combine the generated csv files.  
* plots.py can be utilized to calculate "Control Cost Change in %" vs "Degree of Controllability Plot."