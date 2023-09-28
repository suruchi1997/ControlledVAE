# Controllability-Constrained-Latent-Models
```
git clone https://github.com/suruchi1997/Controllability-Constrained-Latent-Models.git
```
## For Pendulum_Env
 * make a directory conv_mul
   ```
    mkdir conv_mul
   ```  
 * pen_conv.py consists of the training logic. execute pen_conv.py -rs. -rs is the random seed provided for weight initialization. The  base models and controllability-constrained models are trained and saved in the conv_mul/ folder.
   ```
    python pen_conv.py {integer}
   ```  
 * Then mpc_eval2.py is used to execute the evaluation process.

   ```
    python mpc_eval2.py {integer} 
   ```
## For CartPole_Env
1) Make a directory conv_new
2) cp_cont4.py consists of the training logic execute in similar manner as for pendulum envt
3) cp_mpc_eval1.py is used for evaluating the performance of the models
