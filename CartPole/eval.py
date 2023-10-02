import argparse
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
from scipy.optimize import minimize
from train import E2C
from cartpole import ContinuousCartPoleEnv
from PIL import Image
from itertools import repeat
import multiprocessing as mp
import csv


sys.setrecursionlimit(3000)
device = torch.device("cpu")


def split_images(img):
    list_split = []
    for im in img:
        im = im.reshape(1, 80, 160)
        ch, h, w, = im.shape
        w = w // 2
        im = im.squeeze(0)
        im = im[:, w:]

        list_split.append(im.reshape(1, 80, 80))

    return list_split


def min_svd(A,B,O):
    mat1 = torch.stack((B, A @ B,A@A@B,A@A@A@B)).view( 4,4)
    gra1 = (mat1.T@mat1)

    min_svd = torch.min(torch.linalg.svd(gra1)[1])
    max_svd= torch.max(torch.linalg.svd(gra1)[1])
    det1= torch.abs(torch.det(gra1))
    return np.log(min_svd.item()),np.log(max_svd.item()),np.log(det1.item())

# step fucntion
def step_new(x,u,e2c):
    x = torch.tensor(x).float().view(1, 4).to(device)
    h = e2c.trans(x.view(1, 4))

    con = torch.tensor([u]).view(1, 1).float().to(device)
    B_ = e2c.fc_B(h)
    o_ = e2c.fc_o(h)
    A_ = e2c.fc_A(h)
    x = torch.mm(A_.view(4,4), x.view(4, 1)) + torch.mm(B_.view(4, 1), con.view(1, 1)) + o_.view(4, 1)
    if torch.is_tensor(x):
        return x.cpu().detach().numpy()
    else:
        return x

# cost function
def cost_new(u,x_t,z_g,e2c):
    x1 = np.array(x_t).reshape(1,4)
    s=0
    goal=z_g.cpu().detach().numpy().reshape(1, 4)
    s+=0.5*np.sum((goal - x1) ** 2)
    for m in range(len(u)):
        x1=np.array(step_new(x1,u[m],e2c)).reshape(1,4)
        s += 0.5 * np.sum((goal - x1)**2) + 0.5 * (u[m]**2)
    return s

# real cost calculation
def calc_true_cost(state, u, goal):
    return 0.5 * np.linalg.norm(goal[0:4] - state[0:4]), 0.5 * np.sum(u ** 2)

#if angles between -0.86 and 0.86 then sucess otherwise failure
def model_solved(th):
    return all(-0.86<=element <=0.86 for element in th[-5:])


def dyn(z_st,e2c):
    h = e2c.trans(z_st.view(1, 4))
    B_ = e2c.fc_B(h).view(4,1)
    o_ = e2c.fc_o(h).view(4,1)
    A_ = e2c.fc_A(h).view(4,4)
    return A_,B_,o_


def MPC(e2c):
    T = 10
    N = 100
    env = ContinuousCartPoleEnv()
    env.reset_goal()
    raw_img_before = env.render('rgb_array')
    env.step(0.0)
    raw_img_after = env.render('rgb_array')

    img_before = Image.fromarray(raw_img_before).convert('L').resize((80, 80))
    img_after =  Image.fromarray(raw_img_after).convert('L').resize((80, 80))
    img_state =  np.hstack((img_before, img_after))
    goal_1 = torch.tensor(1 - np.array(img_state) / 255.0).view(-1,2,80,80).float().to(device)

    m_g, s_g = e2c.encode(goal_1).chunk(2, dim=1)
    z_g = e2c.reparam(m_g, s_g)

    env.reset_start()
    raw_img_before = env.render('rgb_array')
    env.step(0.0)
    raw_img_after = env.render('rgb_array')
    img_before = Image.fromarray(raw_img_before).convert('L').resize((80, 80))
    img_after = Image.fromarray(raw_img_after).convert('L').resize((80, 80))
    img_state = np.hstack((img_before, img_after))
    img_state_normalized = torch.tensor(1 - np.array(img_state) / 255.0).view(-1,2,80,80).float().to(device)

    m_s, s_s = e2c.encode(img_state_normalized).chunk(2, dim=1)
    z_st=m_s
    x_ = []
    x_.append(np.array([0.1, 0.0]))
    u_ = np.zeros((T,))
    x0 = z_st.cpu().detach().numpy().squeeze()
    th = []
    cost = []
    cons = []
    tup = (-1, 1)
    bnds = tuple(repeat(tup, T))
    di_cost=[]
    ct_cost=[]
    m_svd1=[]
    ma_svd1=[]
    det_1=[]
    plt.ion()
    for s in range(N):
        res = minimize(cost_new, u_, args=(x0,z_g,e2c), method="powell", bounds=bnds,tol=1e-2)
        u_ = res.x
        u0 = u_[0]
        # if optimal action positive apply force of 10N else -10 N
        if u0 < 0:
            st = env.step(-10.0)[0]
        elif u0 > 0:
            st = env.step(10.0)[0]

        raw_img_before = env.render('rgb_array')
        env.step(0.0)
        raw_img_after = env.render('rgb_array')

        img_before = Image.fromarray(raw_img_before).convert('L').resize((80, 80))
        img_after = Image.fromarray(raw_img_after).convert('L').resize((80, 80))
        img_state = np.hstack((img_before, img_after))
        img_state_normalized = torch.tensor(1 - np.array(img_state) / 255.0).view(-1, 2, 80, 80).float().to(device)

        m_s, s_s = e2c.encode(img_state_normalized).chunk(2, dim=1)
        z_st = m_s
        A_,B_,o_=dyn(z_st,e2c)
        m_svd,ma_svd,det1=min_svd(A_,B_,o_)
        m_svd1.append(m_svd)
        ma_svd1.append(ma_svd)
        det_1.append(det1)
        dif_cost, control_cost = calc_true_cost(st, u0, [0,0,0,0])
        di_cost.append(dif_cost)
        ct_cost.append(control_cost)
        x0 = z_st.cpu().detach().numpy().squeeze()
        cons.append(u0)
        cost.append(float(res.fun))
        th.append(st[2])
    return np.mean(cost), np.mean(np.array(ct_cost)), np.mean(np.array(di_cost)), model_solved(np.array(th)),np.mean(np.array(m_svd1)),np.mean(np.array(ma_svd1)),np.mean(np.array(det_1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rs', default=None, type=int, help='Random seed')
    parser.add_argument('--beta', default=None, type=float, help='Beta value')
    args = parser.parse_args()
    e2c = E2C()
    e2c.eval()
    if args.rs is None:
        # find available seeds from the saved models
        r_seeds = [dir_name for dir_name in os.listdir("models")]
        print(f"seed not provided, evaluating seeds in sequence. Available seeds:\n{r_seeds}")
    else:
        r_seeds = [args.rs]

    means = {}
    stds = {}
    trj = 15
    beta_print_once = False
    for r_s in r_seeds:
        if args.beta is None:
            betas = [dir_name[3:-4] for dir_name in os.listdir(f"models/{r_s}") if dir_name.startswith("mod")]
            if not beta_print_once:
                print(f"beta not provided, evaluating betas in sequence. Available betas:\n{betas}")
                beta_print_once = True
        else:
            betas = [args.beta]
        for beta in betas:
            m1 = []
            sd1 = []
            ctrl_cost = []
            diff_cost = []
            m_sv1 = []
            ma_sv1=[]
            de_1=[]
            print(f'{datetime.now().strftime("%H:%M:%S")} eval: started for rs={r_s} beta={beta}')

            mp.set_start_method('spawn', force=True)
            p = mp.Pool()
            e2c = E2C()

            e2c.load_state_dict(torch.load(f"models/{r_s}/mod{beta}.pth", map_location=device))
            cost, control_cost, dif_cost, succ, m_sv,ma_sv,de1 = zip(*p.map(MPC, [e2c] * trj))

            m1.append(cost)
            ctrl_cost.append(control_cost)
            diff_cost.append(dif_cost)
            m_sv1.append(m_sv)
            ma_sv1.append(ma_sv)
            de_1.append(de1)
            p.close()
            p.join()

            per = sum(np.array(succ)) / len(succ)
            print(f"percent solved: {per}")
            csv_exists = os.path.isfile(f"models/{r_s}/eval.csv")
            with open(f"models/{r_s}/eval.csv", 'a', newline='') as f:
                writer1 = csv.writer(f)
                if not csv_exists:
                    writer1.writerow(
                        ["lat_cost_mean", "lat_cost_std", "ctrl_cost", "diff_cost", "success", "beta", "log_min_svd",
                         "ctrl_cost_std", "diff_cost_std", "log_max_svd"])
                writer1.writerow(
                    [np.mean(np.array(m1)), np.std(np.array(m1)), np.mean(np.array(ctrl_cost)), np.mean(np.array(diff_cost)),
                     per, beta, np.mean(np.array(m_sv1)),np.std(np.array(ctrl_cost)),np.std(np.array(diff_cost)),np.mean(np.array(ma_sv1)),])