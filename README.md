# Controllability-Constrained-Latent-Models
```
git clone https://github.com/suruchi1997/Controllability-Constrained-Latent-Models.git
```
* Go to the respective folder
* install required libararies
```
pip install requirements.txt 
```
## For Pendulum_Env
 * make a directory conv_mul
   ```
    mkdir conv_mul
   ```  
 * pen_conv.py consists of the training logic. Execute pen_conv.py -rs. -rs is the random seed provided for weight initialization. The  base models and controllability-constrained models are trained and saved in the conv_mul/ folder.
   ```
    python pen_conv.py {integer}
   ```  
 * Then mpc_eval2.py is used to execute the evaluation process.

   ```
    python mpc_eval2.py {integer} 
   ```
## For CartPole_Env
 * Make a directory conv_new. 
   ```
    mkdir conv_mul
   ```  
 * cp_cont4.py consists of the training logic. Execute cp_cont4.py -rs. -rs is the random seed provided for weight initialization. The  base models and controllability-constrained models are trained and saved in the conv_new/ folder.
   
   ```
    python cp_cont4 .py {integer}
   ```  
 * cp_mpc_eval1.py is used for evaluating the performance of the models

   ```
    python cp_mpc_eval1.py {integer}
   ```
