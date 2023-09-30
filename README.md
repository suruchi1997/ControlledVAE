# Controllability-Constrained-Latent-Models
1. clone the repo:     
```
git clone https://github.com/suruchi1997/Controllability-Constrained-Latent-Models.git
```
2. setup the anaconda environment:       
cuda: `conda env create --file env_cuda.yml`       
cpu: `conda env create --file env_cpu.yml`    
* csv_com.py can be used to combine the csv files   
* plots.py can be utilized to calculate "Control Cost Change in %" vs "Degree of Controllability Plot"    
## For Pendulum_Env
 * pen_conv.py consists of the training logic. Execute pen_conv.py -rs. -rs is the random seed provided for weight initialization. The  base models and controllability-constrained models are trained and saved in the conv_mul/ folder.
   ```
    python pen_conv.py --rs {integer}
   ```  
 * Then mpc_eval2.py is used to execute the evaluation process.

   ```
    python mpc_eval2.py --rs {integer} 
   ```
## For CartPole_Env
 * cp_cont4.py consists of the training logic. Execute cp_cont4.py -rs. -rs is the random seed provided for weight initialization. The  base models and controllability-constrained models are trained and saved in the conv_new/ folder.
   
   ```
    python cp_cont4.py {integer}
   ```  
 * cp_mpc_eval1.py is used for evaluating the performance of the models

   ```
    python cp_mpc_eval1.py {integer}
   ```
