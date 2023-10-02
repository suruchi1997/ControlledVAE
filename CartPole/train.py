import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import gym
from cartpole import ContinuousCartPoleEnv
import argparse
import cv2
import os
from datetime import datetime

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#load saved data
def load_data(t_size):
    c = []
    n = []
    a = []
    with open(f"ac3.txt", "r") as f1:
        for i1 in range(t_size):
            c.append(cv2.imread(os.path.join(os.getcwd(), "curr3", f"cim{str(i1)}.png"), 0).reshape(80, 160))
            n.append(cv2.imread(os.path.join(os.getcwd(), "next3", f"nim{str(i1)}.png"), 0).reshape(80, 160))
            a.append([float(f1.readline()[1:-2])])
    return c, n, a

#save data
def save_data(t_size, c, n, a):
    print("saving data")
    if not os.path.exists("curr3"):
        os.mkdir("curr3")
    if not os.path.exists("next3"):
        os.mkdir("next3")
    with open(f"ac3.txt", "w") as f1:
        for i1 in range(t_size):
            cv2.imwrite(os.path.join(os.getcwd(), "curr3", f"cim{str(i1)}.png"), c[i1])
            cv2.imwrite(os.path.join(os.getcwd(), "next3", f"nim{str(i1)}.png"), n[i1])
            f1.write(str(a[i1]) + "\n")
    print("saved")

# collect data
def data_coll(t_size,use_cached=True):
    if use_cached and os.path.exists(f"ac3.txt") and os.path.exists("curr3") and os.path.exists("next3"):
        print("loading cached data")
        return load_data(t_size)
    env = ContinuousCartPoleEnv()
    c_st = []
    n_st = []
    ac = []

    angs=np.random.uniform(-np.pi/2,np.pi/2,size=t_size)
    for i,j in enumerate(angs):
        env.reset(j)
        img1 = env.render("rgb_array")
        env.step(0.0)
        img2 = env.render("rgb_array")
        a2 = np.random.uniform(-1,1)
        env.step(a2)

        img3 = env.render("rgb_array")
        env.step(0.0)
        img4=env.render("rgb_array")
        im1 = Image.fromarray(img1).convert('L').resize((80, 80))
        im2 = Image.fromarray(img2).convert('L').resize((80, 80))
        im3 = np.hstack((im1, im2))

        c_st.append(im3)

        im4 = Image.fromarray(img3).convert('L').resize((80, 80))
        im5= Image.fromarray(img4).convert('L').resize((80, 80))
        im6 = np.hstack((im4, im5))
        n_st.append(im6)
        ac.append([a2])

    save_data(t_size,c_st,n_st,ac)
    return c_st, n_st, ac

#orthogoanl intialization of weights
def weights_init(m):
    if type(m) in [nn.Conv2d, nn.Linear, nn.ConvTranspose2d]:
        torch.nn.init.orthogonal_(m.weight)

# computation of total loss
# recon_term - (x_curr,x_reconstructed)
# pred_term - (x_next , x_predicted_by_trans)
def compute_loss(x, x_next, q_z_next, x_recon, x_next_pred, q_z, q_z_next_pred, min_svd_mean,batch_size, lamda,beta):
    # lower-bound loss
    recon_term = -torch.mean(torch.sum(x.view(batch_size, 12800) * torch.log(1e-5 + x_recon)
                                       + (1 - x.view(batch_size, 12800)) * torch.log(1e-5 + 1 - x_recon), dim=1))
    pred_loss = -torch.mean(torch.sum(x_next.view(batch_size, 12800) * torch.log(1e-5 + x_next_pred)
                                      + (1 - x_next.view(batch_size, 12800)) * torch.log(1e-5 + 1 - x_next_pred),
                                      dim=1))

    kl_term = - 0.5 * torch.mean(torch.sum(1 + q_z.logvar - q_z.mean.pow(2) - q_z.logvar.exp(), dim=1))
    lower_bound = recon_term + pred_loss + kl_term
    consis_term = NormalDistribution.KL_divergence(q_z_next_pred, q_z_next)

    return lower_bound + lamda * consis_term - beta* min_svd_mean

#Normal Distribution  return cov matrix
def NormDist(mean, logvar, A=None):
    sigma = torch.diag_embed(torch.exp(logvar))
    if A is None:
        cov = sigma
    else:
        cov = A.bmm(sigma.bmm(A.transpose(1, 2)))
    return cov

# Normal Distribution to store mean and variance
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
        t1 = torch.distributions.Normal(q_z_next.mean, (q_z_next.logvar / 2).exp())
        t2 = torch.distributions.Normal(q_z_next_pred.mean, (q_z_next_pred.logvar / 2).exp())
        KL = torch.distributions.kl_divergence(t1,t2).mean()
        return KL


class E2C(nn.Module):
    def __init__(self):
        super(E2C, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(10 * 10 * 10, 200),
            nn.ReLU(),
            nn.Linear(200, 4 * 2))
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(4, 200),
            nn.ReLU(),
            nn.Linear(200, 1000),
            nn.ReLU(),
            nn.Unflatten(-1, ([10, 10, 10])),
            nn.ConvTranspose2d(in_channels=10, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.Upsample(scale_factor=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.Upsample(scale_factor=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.Upsample(scale_factor=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=2, kernel_size=5, stride=1, padding=2),
            nn.Flatten(),
            nn.Sigmoid()
        )
        # transition net
        self.trans = nn.Sequential(
            nn.Linear(4, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
        )
        self.fc_A = nn.Sequential(
            nn.Linear(100, 4 * 4),  # v_t and r_t
        )
        self.fc_A.apply(weights_init)
        self.trans.apply(weights_init)
        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)

        self.fc_B = nn.Linear(100, 4 * 1)
        torch.nn.init.orthogonal_(self.fc_B.weight)
        self.fc_o = nn.Linear(100, 4)
        torch.nn.init.orthogonal_(self.fc_o.weight)

    # transition function to compute the grammian
    def transition(self, z_bar_t, q_z_t, u_t,batch_size):
        h_t = self.trans(z_bar_t)
        B_t = self.fc_B(h_t)
        o_t = self.fc_o(h_t)
        A_t = self.fc_A(h_t)

        B_t = B_t.view(-1, 4, 1)
        A_t = A_t.view(-1, 4, 4)
        mu_t = q_z_t.mean

        mean = A_t.bmm(mu_t.unsqueeze(-1)).squeeze(-1) + B_t.bmm(u_t.unsqueeze(-1)).squeeze(-1) + o_t
        # C=[B,AB,A^2B,A^3B]
        mat1 = torch.stack((B_t,
                            A_t@B_t,
                            A_t@A_t@B_t,
                            A_t@A_t@A_t@B_t
                            ),dim=1).view(batch_size, 4,4)

        #Grammian = CC'
        gra1 = (mat1 @ mat1.permute(0, 2, 1)) + 0.000 * torch.eye(4).repeat(batch_size, 1, 1).to(device)

        if batch_size>0:
            min_svd_mean = torch.mean(torch.log(torch.min(torch.linalg.svd(gra1)[1], dim=1).values))
        else:
            min_svd_mean = torch.log(torch.min(torch.linalg.svd(gra1)[1], dim=1).values)

        return mean, NormalDistribution(mean, logvar=q_z_t.logvar, A=A_t), min_svd_mean

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def reparam(self, mean, logvar):
        sigma = (logvar / 2).exp()
        epsilon = torch.randn_like(sigma)
        return mean + torch.mul(epsilon, sigma)

    def forward(self, x, u, x_next,batch_size):
        mu, logvar = self.encode(x).chunk(2, dim=1)
        z = self.reparam(mu, logvar)
        q_z = NormalDistribution(mu, logvar)
        x_recon = self.decode(z)

        z_next, q_z_next_pred, min_svd_mean = self.transition(z, q_z, u,batch_size)

        x_next_pred = self.decode(z_next)

        mu_next, logvar_next, = self.encode(x_next).chunk(2, dim=1)
        q_z_next = NormalDistribution(mean=mu_next, logvar=logvar_next)

        return x_recon, x_next_pred, q_z, q_z_next_pred, q_z_next, min_svd_mean

    def predict(self, x, u):
        mu, logvar = self.encoder(x)
        z = self.reparam(mu, logvar)
        q_z = NormalDistribution(mu, logvar)

        z_next, q_z_next_pred, _ = self.transition(z, q_z, u)

        x_next_pred = self.decode(z_next)
        return x_next_pred

# train function
def train(
        batch_size=64,
        epochs=55000,
        training_size=15000,
        lr=0.0001,
        beta=0.05,
        seed=0,
        logging_interval=2000,
        train_base_mod=False,
        skip_trained=False
):
    gym.logger.set_level(40)
    curr, next, ac = data_coll(training_size)

    c1 = torch.tensor(1 - (np.array(curr)) / 255.0).view(-1, 2, 80, 80).float().to(device)
    n1 = torch.tensor(1 - (np.array(next)) / 255.0).view(-1, 2, 80, 80).float().to(device)
    a1 = torch.tensor(np.array(ac)).float().to(device)
    e2c = E2C().to(device)
    optimizer = optim.Adam(e2c.parameters(), lr=lr)

    # base model has to be learned first
    if not os.path.exists(f"models/{seed}/base_mod.pth") and not train_base_mod:
        print(f"base model not found for seed {seed}, training base model")
        train(batch_size=batch_size, epochs=70000, training_size=training_size, lr=lr, beta=0.0, train_base_mod=True, seed=seed)

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
        x_recon, x_next_pred, q_z, q_z_next_pred, q_z_next, min_svd_mean = e2c(x_t1, u_t1, x_p1,batch_size)

        total = compute_loss(x_t1, x_p1, q_z_next, x_recon, x_next_pred, q_z, q_z_next_pred, min_svd_mean,batch_size, 1.0,beta)

        if (i+1)%logging_interval==0:
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


if __name__ == '__main__':
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    env = ContinuousCartPoleEnv()
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