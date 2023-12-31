from datetime import datetime

import torch
import torch.nn as nn
from pendulum1 import PendulumEnv1
import numpy as np
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import gym
import cv2
import os
import argparse


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#load saved data
def load_data(t_size):
    c = []
    n = []
    a = []
    with open(f"ac.txt", "r") as f1:
        for i1 in range(t_size):
            c.append(cv2.imread(os.path.join(os.getcwd(), "curr", f"cim{str(i1)}.png"), 0).reshape(48, 96))
            n.append(cv2.imread(os.path.join(os.getcwd(), "next", f"nim{str(i1)}.png"), 0).reshape(48, 96))
            a.append([float(f1.readline()[1:-2])])
    return c, n, a

#save data
def save_data(t_size, c, n, a):
    print("saving data")
    if not os.path.exists("curr"):
        os.mkdir("curr")
    if not os.path.exists("next"):
        os.mkdir("next")
    with open(f"ac.txt", "w") as f1:
        for i1 in range(t_size):
            cv2.imwrite(os.path.join(os.getcwd(), "curr", f"cim{str(i1)}.png"), c[i1])
            cv2.imwrite(os.path.join(os.getcwd(), "next", f"nim{str(i1)}.png"), n[i1])
            f1.write(str(a[i1]) + "\n")
    print("saved")

# collect random samples
#use_cached = True helps to utilize the cached dataset
def data_coll(t_size, env, use_cached=True):
    if use_cached and os.path.exists(f"ac.txt") and os.path.exists("curr") and os.path.exists("next"):
        print("loading cached data")
        return load_data(t_size)
    print("generating data")
    c_st = []
    n_st = []
    ac = []
    for i in range(t_size):
        th = np.random.uniform(-np.pi, np.pi)
        thdot = np.random.uniform(-10, 10)
        state = np.array([th, thdot])
        before1 = state
        before2 = env.step_from_state(state, [0])

        _, im1 = cv2.threshold(
            cv2.cvtColor(cv2.resize(env.render_state(before1, "rgb_array"), (48, 48)), cv2.COLOR_BGR2GRAY), 128, 255,
            cv2.THRESH_BINARY)

        _, im2 = cv2.threshold(
            cv2.cvtColor(cv2.resize(env.render_state(before2, "rgb_array"), (48, 48)), cv2.COLOR_BGR2GRAY), 128, 255,
            cv2.THRESH_BINARY)
        im3 = np.hstack((im1, im2))

        c_st.append(im3)

        u = np.random.uniform(-2, 2, size=(1,))
        state = env.step_from_state(state, u)
        after1 = state
        after2 = env.step_from_state(state, [0])

        _, im4 = cv2.threshold(
            cv2.cvtColor(cv2.resize(env.render_state(after1, "rgb_array"), (48, 48)), cv2.COLOR_BGR2GRAY), 128, 255,
            cv2.THRESH_BINARY)

        _, im5 = cv2.threshold(
            cv2.cvtColor(cv2.resize(env.render_state(after2, "rgb_array"), (48, 48)), cv2.COLOR_BGR2GRAY), 128, 255,
            cv2.THRESH_BINARY)
        im6 = np.hstack((im4, im5))

        n_st.append(im6)
        ac.append(u)

    save_data(t_size, c_st,n_st,ac)
    return c_st, n_st, ac

# orthogonal initialization of weight
def weights_init(m):
    if type(m) in [nn.Conv2d, nn.Linear, nn.ConvTranspose2d]:
        torch.nn.init.orthogonal_(m.weight)

# computing overall loss
# recon_term - (x_curr,x_reconstructed)
# pred_term - (x_next , x_predicted_by_trans)
def compute_loss(x, x_next, q_z_next, x_recon, x_next_pred, q_z, q_z_next_pred, min_svd_mean, lamda, beta=0.05,
                 batch_size=256):
    recon_term = -torch.mean(torch.sum(x.view(batch_size, 4608) * torch.log(1e-5 + x_recon)
                                       + (1 - x.view(batch_size, 4608)) * torch.log(1e-5 + 1 - x_recon), dim=1))
    pred_loss = -torch.mean(torch.sum(x_next.view(batch_size, 4608) * torch.log(1e-5 + x_next_pred)
                                      + (1 - x_next.view(batch_size, 4608)) * torch.log(1e-5 + 1 - x_next_pred), dim=1))

    kl_term = - 0.5 * torch.mean(torch.sum(1 + q_z.logvar - q_z.mean.pow(2) - q_z.logvar.exp(), dim=1))

    lower_bound = recon_term + pred_loss + kl_term - beta * min_svd_mean
    consis_term = NormalDistribution.KL_divergence(q_z_next_pred, q_z_next)
    return lower_bound + lamda * consis_term

# Normal distribution to return covariance
def NormDist(mean, logvar, A=None):
    sigma = torch.diag_embed(torch.exp(logvar))
    if A is None:
        cov = sigma
    else:
        cov = A.bmm(sigma.bmm(A.transpose(1, 2)))
    return cov

#Normal Distribution to save mean and variance
class NormalDistribution:
    def __init__(self, mean, logvar, A=None):
        self.mean = mean
        self.logvar = logvar
        self.A = A
        sigma = torch.diag_embed(torch.exp(logvar))
        if A is None:
            self.cov = sigma
        else:
            self.cov = A.bmm(sigma.bmm(A.transpose(1, 2)))

    @staticmethod
    # KL divergence between (z_hat_t+1,z_t+1)
    def KL_divergence(q_z_next_pred, q_z_next):
        cov1 = NormDist(q_z_next.mean, q_z_next.logvar)
        cov2 = NormDist(q_z_next_pred.mean, q_z_next_pred.logvar, q_z_next_pred.A)
        mu_0 = q_z_next_pred.mean
        mu_1 = q_z_next.mean
        sigma_0 = torch.exp(q_z_next_pred.logvar)
        sigma_1 = torch.exp(q_z_next.logvar)

        k = float(q_z_next_pred.mean.size(1))
        d1 = torch.linalg.det(cov1)
        d2 = torch.linalg.det(cov2)
        d = torch.abs((d1 / d2))
        sum = lambda x: torch.sum(x, dim=1)

        KL = 0.5 * torch.mean(
            sum(torch.diagonal(torch.bmm(torch.linalg.inv(cov1), cov2), dim1=-1, dim2=-2))
          + sum(torch.pow(mu_1 - mu_0, 2) / sigma_1) - k
          + torch.log(d)
        )
        t1 = torch.distributions.Normal(q_z_next.mean, (q_z_next.logvar / 2).exp())
        t2 = torch.distributions.Normal(q_z_next_pred.mean, (q_z_next_pred.logvar / 2).exp())
        KL = torch.distributions.kl_divergence(t1, t2).mean()
        return KL


class E2C(nn.Module):
    def __init__(self, device='cpu'):
        super(E2C, self).__init__()
        #encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*12*12, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * 2),
        ).to(device)
        #decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Linear(256, 32 * 12 * 12),
            nn.ReLU(),
            nn.Unflatten(-1, (32, 12, 12)),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Flatten(),
            nn.Sigmoid()
        ).to(device)

        #transition network
        self.trans = nn.Sequential(
            nn.Linear(2, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
        ).to(device)
        self.fc_A = nn.Sequential(
            nn.Linear(100, 2 * 2),  # v_t and r_t
        ).to(device)
        self.fc_A.apply(weights_init)
        self.trans.apply(weights_init)
        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)

        self.fc_B = nn.Linear(100, 2 * 1).to(device)
        torch.nn.init.orthogonal_(self.fc_B.weight)
        self.fc_o = nn.Linear(100, 2).to(device)
        torch.nn.init.orthogonal_(self.fc_o.weight)

    # transition function to compute grammian
    def transition(self, z_bar_t, q_z_t, u_t):
        batch_size = z_bar_t.size(0)
        h_t = self.trans(z_bar_t)
        B_t = self.fc_B(h_t)
        o_t = self.fc_o(h_t)
        A_t = self.fc_A(h_t).view(-1, 2, 2)
        B_t = B_t.view(-1, 2, 1)

        mu_t = q_z_t.mean
        mean = A_t.bmm(mu_t.unsqueeze(-1)).squeeze(-1) + B_t.bmm(u_t.unsqueeze(-1)).squeeze(-1) + o_t
        #C=[B,AB]
        mat1 = torch.stack((B_t, torch.bmm(A_t, B_t)), dim=1).view(batch_size, 2, 2)
        #grammian = CC'
        gra1 = (mat1@mat1.permute(0, 2, 1)) + 0.000 * torch.eye(2).repeat(batch_size, 1, 1).to(device)
        min_svd_mean = torch.log(torch.min(torch.linalg.svd(gra1)[1], dim=1).values)

        if batch_size > 0:
            svd_mean = torch.mean(min_svd_mean)
        return mean, NormalDistribution(mean, logvar=q_z_t.logvar, A=A_t), svd_mean

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def reparam(self, mean, logvar):
        sigma = (logvar / 2).exp()
        epsilon = torch.randn_like(sigma)
        return mean + torch.mul(epsilon, sigma)

    def forward(self, x, u, x_next):
        mu, logvar = self.encode(x).chunk(2, dim=1)
        z = self.reparam(mu, logvar)
        q_z = NormalDistribution(mu, logvar)
        x_recon = self.decode(z)

        z_next, q_z_next_pred, min_svd_mean = self.transition(z, q_z, u)

        x_next_pred = self.decode(z_next)
        mu_next, logvar_next = self.encode(x_next).chunk(2, dim=1)
        q_z_next = NormalDistribution(mean=mu_next, logvar=logvar_next)

        return x_recon, x_next_pred, q_z, q_z_next_pred, q_z_next, min_svd_mean

# train starts
def train(
        batch_size=256,
        epochs=150000,
        training_size=15000,
        lr=0.0003,
        beta=0.05,
        eval_freq=5000,
        train_base_mod=False,
        seed=0,
        skip_trained=False,
):
    """
    trains either a base model or a controllability model.
    """
    gym.logger.set_level(40)
    env = PendulumEnv1()

    curr, next, ac = data_coll(training_size, env)
    c1 = torch.tensor(1 - (np.array(curr)) / 255.0).view(-1, 2, 48, 48).float().to(device)
    n1 = torch.tensor(1 - (np.array(next)) / 255.0).view(-1, 2, 48, 48).float().to(device)
    a1 = torch.tensor(np.array(ac)).float().to(device)
    e2c = E2C(device).to(device)
    optimizer = optim.Adam(e2c.parameters(), lr=lr)

    # base model has to be learned first
    if not os.path.exists(f"models/{seed}/base_mod.pth") and not train_base_mod:
        print(f"base model not found for seed {seed}, training base model")
        train(batch_size=batch_size, epochs=100000, training_size=training_size, lr=lr, beta=0.0, train_base_mod=True, seed=seed)

    if os.path.exists(f"models/{seed}/mod{beta}.pth") and skip_trained:
        print(f"model with beta={beta} already trained for seed {seed}, skipping")
        return

    if not train_base_mod:
        e2c.load_state_dict(torch.load(f"models/{seed}/base_mod.pth", map_location=device))
        print(f"training model with beta={beta}")

    for i in range(epochs):
        e2c.train()

        start_idx = (batch_size * i) % training_size
        if start_idx + batch_size >= training_size:
            start_idx = random.randint(0, training_size - batch_size)

        x_t1 = c1[start_idx:start_idx + batch_size]
        u_t1 = a1[start_idx:start_idx + batch_size]
        x_p1 = n1[start_idx:start_idx + batch_size]
        x_recon, x_next_pred, q_z, q_z_next_pred, q_z_next, min_svd_mean = e2c(x_t1, u_t1, x_p1)
        total = compute_loss(
            x_t1, x_p1, q_z_next, x_recon, x_next_pred, q_z, q_z_next_pred, min_svd_mean,0.45,
            beta=beta, batch_size=batch_size
        )
        if (i+1)%eval_freq==0:
            print(f'{datetime.now().strftime("%H:%M:%S")} epoch {i+1}/{epochs} loss: {total.item()}')

        optimizer.zero_grad()
        total.backward()
        optimizer.step()

    if not os.path.exists(f"models/{seed}"):
        os.makedirs(f"models/{seed}")
    if not train_base_mod:
        torch.save(e2c.state_dict(), f"models/{seed}/mod{beta}.pth")
    else:
        torch.save(e2c.state_dict(), f"models/{seed}/base_mod.pth")


if __name__ == "__main__":
    # to disable the display during data collection
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    parser = argparse.ArgumentParser()
    parser.add_argument('--rs',   default=None, type=int, help='Random seed')
    parser.add_argument('--beta', default=None, type=float, help='Beta value')
    parser.add_argument('--skip-trained', default=True, type=float, help='If true, skip training if model exists')
    args = parser.parse_args()
    if args.rs is None:
        with open('rseeds.txt', 'r') as f:
            seeds = [int(n) for n in f.readline().split(',')]
        print(f"seed not provided, training seeds in sequence. Available seeds:\n{seeds}")
    else:
        seeds = [args.rs]
    beta_print_once = False
    for seed in seeds:
        torch.manual_seed(seed)
        if args.beta is not None:
            beta = args.beta
            train(beta=beta, seed=seed, skip_trained=args.skip_trained)
        else:
            with open('betas.txt', 'r') as f:
                betas = [float(n) for n in f.readline().split(',')]
            if not beta_print_once:
                print(f"beta not provided, training betas in sequence. Available betas:\n{betas}")
                beta_print_once = True
            for beta in betas:
                train(beta=beta, seed=seed, skip_trained=args.skip_trained)