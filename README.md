# Controllability-Constrained-Latent-Models
## Pendulum_Env
 1) make a directory conv_mul
 2) pen_conv.py consists of the training logic. execute pen_conv.py -rs. -rs is the random seed provided for weight initialization. The  base models and controllability-constrained models are trained and saved in the conv_mul/ folder.
 3) Then mpc_eval2.py is used to execute the evaluation process.
## CartPole_Env
1) Make a directory conv_mul1
2) cp_cont4.py consists of the training logic execute in similar manner as for pendulum envt
3) cp_mpc_eval1.py is used for evaluating the performance of the models
