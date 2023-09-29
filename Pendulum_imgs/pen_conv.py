import torch
import torch.nn as nn
from PIL import Image
from pendulum1 import PendulumEnv1
import numpy as np
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import gym
import cv2
import os


os.environ['SDL_VIDEODRIVER'] = 'dummy'
device = torch.device("cpu")
import argparse

parser = argparse.ArgumentParser(description='My Program')
parser.add_argument('rs', help='Random seed')
args = parser.parse_args()
seed1=int(float(args.rs))
torch.manual_seed(seed1)

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

#orthogonal initialization of weight
def weights_init(m):
    if type(m) in [nn.Conv2d, nn.Linear, nn.ConvTranspose2d]:

        torch.nn.init.orthogonal_(m.weight)



# computing overall loss
# recon_term - (x_curr,x_reconstructed)
#pred_term - (x_next , x_predicted_by_trans)
def compute_loss(x, x_next, q_z_next, x_recon, x_next_pred, q_z, q_z_next_pred, min_svd_mean, lamda, beta=0.05,
                 batch_size=256):
    recon_term = -torch.mean(torch.sum(x.view(batch_size, 4608) * torch.log(1e-5 + x_recon)
                                       + (1 - x.view(batch_size, 4608)) * torch.log(1e-5 + 1 - x_recon), dim=1))
    pred_loss = -torch.mean(torch.sum(x_next.view(batch_size, 4608) * torch.log(1e-5 + x_next_pred)
                                      + (1 - x_next.view(batch_size, 4608)) * torch.log(1e-5 + 1 - x_next_pred), dim=1))

    kl_term = - 0.5 * torch.mean(torch.sum(1 + q_z.logvar - q_z.mean.pow(2) - q_z.logvar.exp(), dim=1))

    lower_bound = recon_term + pred_loss + kl_term - beta * min_svd_mean
    consis_term = NormalDistribution.KL_divergence(q_z_next_pred, q_z_next)
    pred1.append(pred_loss.item())
    rec1.append(recon_term.item())
    kl1.append(kl_term.item())
    consi1.append(consis_term.item())
    min_eig.append(min_svd_mean.item())
    return lower_bound + lamda * consis_term, [recon_term, pred_loss, kl_term, consis_term]

# Normal distribution to return covariance
def NormDist(mean, logvar, A=None):
    mean = mean
    logvar = logvar

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

        KL = 0.5 * torch.mean(sum(torch.diagonal(torch.bmm(torch.linalg.inv(cov1), cov2), dim1=-1, dim2=-2))
                              + sum(torch.pow(mu_1 - mu_0, 2) / sigma_1) - k
                              + torch.log(d)
                              )
        t1 = torch.distributions.Normal(q_z_next.mean, (q_z_next.logvar / 2).exp())
        t2 = torch.distributions.Normal(q_z_next_pred.mean, (q_z_next_pred.logvar / 2).exp())
        KL = torch.distributions.kl_divergence(t1, t2).mean()
        return KL


class E2C(nn.Module):
    def __init__(self, device='cpu', minimize_log_det_gram=True, minimize_svd=False):
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
        self.minimize_svd = minimize_svd
        self.minimize_log_det_gram = minimize_log_det_gram

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

    #returns  encoded input
    def encode(self, x):

        return self.encoder(x)
    # returns decoded output
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
        weight_decay=0.002,
        beta=0.05,
        model_path="model",
        f_eval=None,
        f_eval_args=(None,),
        eval_freq=5000,
        minimize_svd_grm=False,
        minimize_log_det_gram=False,
        debug_vis=False,
        base_mod=False
):
    gym.logger.set_level(40)


    env = PendulumEnv1()

    curr, next, ac = data_coll(training_size, env)


    c1 = torch.tensor(1 - (np.array(curr)) / 255.0).view(-1, 2, 48, 48).float().to(device)

    n1 = torch.tensor(1 - (np.array(next)) / 255.0).view(-1, 2, 48, 48).float().to(device)

    a1 = torch.tensor(np.array(ac)).float().to(device)

    e2c = E2C(device, minimize_svd_grm, minimize_log_det_gram).to(device)

    optimizer = optim.Adam(e2c.parameters(), lr=lr)

    if debug_vis:
        plt.ion()
        deb_losses = np.zeros((epochs, 4))

    if base_mod:
        epochs=100000
    if not base_mod:

        e2c.load_state_dict(torch.load("conv_mul/conv"+str(seed1)+"/base_mod.pth", map_location=device))

    for i in range(epochs):

        e2c.train()

        start_idx = (batch_size * i) % training_size

        if start_idx + batch_size >= training_size:
            start_idx = random.randint(0, training_size - batch_size)

        x_t1 = c1[start_idx:start_idx + batch_size]

        u_t1 = a1[start_idx:start_idx + batch_size]

        x_p1 = n1[start_idx:start_idx + batch_size]
        x_recon, x_next_pred, q_z, q_z_next_pred, q_z_next, min_svd_mean = e2c(x_t1, u_t1, x_p1)
        total, losses = compute_loss(x_t1, x_p1, q_z_next, x_recon, x_next_pred, q_z, q_z_next_pred, min_svd_mean,0.45,
                                     beta=beta, batch_size=batch_size)


        if i % eval_freq == 0:
            print(start_idx, total)

        optimizer.zero_grad()
        total.backward()
        optimizer.step()
    if not base_mod:
        torch.save(e2c.state_dict(), "conv_mul/conv"+str(seed1)+"/mod" + str(beta) + ".pth")
    else:
        if not os.path.exists("conv_mul/conv"+str(seed1)):
            print("make")
            os.mkdir("conv_mul/conv"+str(seed1))
        torch.save(e2c.state_dict(), "conv_mul/conv"+str(seed1)+"/base_mod.pth")

    # plotting loss components
    if debug_vis:
        plt.subplot(321)
        plt.plot(rec1)
        plt.title("recon")

        plt.subplot(322)
        plt.plot(pred1)
        plt.title("pred")

        plt.subplot(323)
        plt.plot(kl1)
        plt.title("kl_div")

        plt.subplot(324)
        plt.plot(consi1)
        plt.title("consis")

        plt.subplot(326)
        plt.plot(min_eig)
        plt.title("min_svd_grammain")
        if not base_mod:
            plt.savefig("conv_mul/conv"+str(seed1)+"/plot" + str(beta) + ".png")
            plt.show()


if __name__ == "__main__":
    betas=[0.0,0.0005,0.005,0.05,0.5,0.7,0.9,1.0,5.0]
    for b in betas:
        min_eig = []
        rec1 = []
        pred1 = []
        kl1 = []
        consi1 = []
        # start to train using base_mod
        if b=="base_mod":
            train(beta=0, minimize_svd_grm=True, debug_vis=True,base_mod=True)  # 0.05
        else:
            train(beta=b, minimize_svd_grm=True, debug_vis=True,base_mod=False)
