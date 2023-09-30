import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
from scipy.optimize import minimize
from pen_conv import E2C
from pendulum1 import PendulumEnv1
from torch.utils.tensorboard import SummaryWriter
from itertools import repeat
import multiprocessing as mp
import cv2
import csv


sys.setrecursionlimit(3000)
device = torch.device("cpu")


def split_images(img):
    list_split = []
    for im in img:
        im = im.reshape(1, 48, 96)
        ch, h, w, = im.shape
        w = w // 2
        im = im.squeeze(0)
        im = im[:, w:]
        list_split.append(im.reshape(1, 48, 48))
    return list_split


def min_svd(A,B,O):
    mat1 = torch.stack((B, A @ B)).view( 2, 2)
    gra1 = (mat1.T@mat1)
    min_svd = torch.min(torch.linalg.svd(gra1)[1])
    max_svd= torch.max(torch.linalg.svd(gra1)[1])
    
    return np.log(min_svd.item()),np.log(max_svd.item())

# step function
def step_new(x,u,e2c):
    x = torch.tensor(x).float().view(1, 2).to(device)
    h = e2c.trans(x.view(1, 2))
    con = torch.tensor([u]).view(1, 1).float().to(device)
    B_ = e2c.fc_B(h)
    o_ = e2c.fc_o(h)
    A_ = e2c.fc_A(h)
    x = torch.mm(A_.view(2, 2), x.view(2, 1)) + torch.mm(B_.view(2, 1), con.view(1, 1)) + o_.view(2, 1)

    if torch.is_tensor(x):
        return x.cpu().detach().numpy()
    else:
        return x

#cost function
def cost_new(u,x_t,z_g,e2c):
    x1 = np.array(x_t).reshape(1,2)
    s=0
    goal=z_g.cpu().detach().numpy().reshape(1, 2)
    s+=0.5*np.sum((goal - x1) ** 2)
    for m in range(len(u)):
        x1=np.array(step_new(x1,u[m],e2c)).reshape(1,2)
        s += 0.5 * np.sum((goal - x1)**2) + 0.5 * (u[m]**2 *0.1)
    return s

# real cost calculation
def calc_true_cost(state, u, goal):
    if state.shape[0] == 2:
        theta, thetadot = state
        state = np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)
        state= np.array([theta,thetadot])
    if goal.shape[0] == 2:
        theta, thetadot = goal

        goal=np.array([theta,thetadot])
    return 0.5 * np.linalg.norm(goal[0:2] - state[0:2]), 0.05 * np.sum(u ** 2)

#if angles for last 5 steos  between -0.4 and 0.4 then sucess otherwise failure
def model_solved(th):
    return all(-0.4<=element <=0.4 for element in th[-5:])


def dyn(z_st,e2c):
    h = e2c.trans(z_st.view(1, 2))

    B_ = e2c.fc_B(h).view(2,1)
    o_ = e2c.fc_o(h).view(2,1)
    A_ = e2c.fc_A(h).view(2,2)
    return A_,B_,o_


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def MPC(e2c):
    state = np.array([0.0, 0.0])
    T = 10
    N = 120
    goal = np.array([0.0, 0.0])
    env = PendulumEnv1()
    before1 = state
    before2 = env.step_from_state(state, np.array([0]))

    _, im1 = cv2.threshold(
        cv2.cvtColor(cv2.resize(env.render_state(before1, "rgb_array"), (48, 48)), cv2.COLOR_BGR2GRAY), 128, 255,
        cv2.THRESH_BINARY)

    _, im2 = cv2.threshold(
        cv2.cvtColor(cv2.resize(env.render_state(before2, "rgb_array"), (48, 48)), cv2.COLOR_BGR2GRAY), 128, 255,
        cv2.THRESH_BINARY)


    im3 = np.hstack((im1, im2))
    goal_1 = torch.tensor(1 - np.array(im3) / 255.0).view(-1,2,48,48).float().to(device)

    m_g, s_g = e2c.encode(goal_1).chunk(2, dim=1)
    z_g=m_g

    # start state for mpc_evaluation
    state =np.array([np.random.uniform(np.pi-0.5,np.pi+0.5),0.0])

    before1 = state
    before2 = env.step_from_state(state, np.array([0]))

    _, im1 = cv2.threshold(
        cv2.cvtColor(cv2.resize(env.render_state(before1, "rgb_array"), (48, 48)), cv2.COLOR_BGR2GRAY), 128, 255,
        cv2.THRESH_BINARY)

    _, im2 = cv2.threshold(
        cv2.cvtColor(cv2.resize(env.render_state(before2, "rgb_array"), (48,48)), cv2.COLOR_BGR2GRAY), 128, 255,
        cv2.THRESH_BINARY)


    im3 = np.hstack((im1, im2))
    state1 = torch.tensor(1 - np.array(im3) / 255.0).view(-1,2,48,48).float().to(device)

    m_s, s_s = e2c.encode(state1).chunk(2, dim=1)
    z_st=m_s
    x_ = []

    x_.append(np.array([0.1, 0.0]))
    u_ = np.zeros((T, 1))

    x0 = z_st.cpu().detach().numpy()

    th = []
    th_dot = []
    cost = []
    cons = []

    tup = (-2, 2)
    bnds = tuple(repeat(tup, T))
    di_cost=[]
    ct_cost=[]
    m_svd1=[]
    ma_svd1=[]
    plt.ion()
    vis = plt.figure()
    pred = plt.figure().add_subplot()
    for s in range(N):
        res = minimize(cost_new, u_, args=(x0,z_g,e2c), method="powell", bounds=bnds)
        u_ = res.x.reshape(T, 1)
        u0 = u_[0]

        st=np.copy(state)
        th.append(angle_normalize(st[0].item()))
        th_dot.append(state[1])

        before1 = env.step_from_state(state, np.array([u0]))
        before2 = env.step_from_state(before1, np.array([0.0]))

        _, im1 = cv2.threshold(
            cv2.cvtColor(cv2.resize(env.render_state(before1, "rgb_array"), (48, 48)), cv2.COLOR_BGR2GRAY), 128, 255,
            cv2.THRESH_BINARY)

        _, im2 = cv2.threshold(
            cv2.cvtColor(cv2.resize(env.render_state(before2, "rgb_array"), (48, 48)), cv2.COLOR_BGR2GRAY), 128, 255,
            cv2.THRESH_BINARY)

        im3 = np.hstack((im1, im2))
        state1 = torch.tensor(1 - np.array(im3) / 255.0).view(-1, 2,48,48).float().to(device)

        m_s, s_s = e2c.encode(state1).chunk(2, dim=1)
        z_st=m_s
        A_,B_,o_=dyn(z_st,e2c)
        m_svd,ma_svd=min_svd(A_,B_,o_)
        m_svd1.append(m_svd)
        ma_svd1.append(ma_svd)
        state = before1
        state1=np.array([angle_normalize(state[0]),state[1]])
        dif_cost, control_cost = calc_true_cost(state1, u0, goal)
        di_cost.append(dif_cost)
        ct_cost.append(control_cost)
        x0 = z_st.cpu().detach().numpy()
        cons.append(u0)
        cost.append(float(res.fun))

        vis.clear()
        vis_creal = vis.add_subplot()
        vis_cpred = vis_creal.twinx()
        plt.show()
        # pred.clear()
        pred.set_title(f"real step {s},predicted cost {res.fun:.2f},angle {angle_normalize(state[0])}")
        pred.imshow(e2c.decode(z_st).cpu().detach().numpy().reshape(48,96),cmap="gray")

        vis_creal.legend()
        vis_cpred.legend()
        vis.tight_layout()
    return np.mean(cost), np.mean(np.array(ct_cost)), np.mean(np.array(di_cost)), model_solved(np.array(th)),np.mean(np.array(m_svd1)),np.mean(np.array(ma_svd1))


if __name__ == '__main__':
    e2c = E2C()
    writer = SummaryWriter()
    e2c.eval()

    betas=[0.0,0.0005,0.005,0.05,0.5,0.7,0.9,1.0,5.0]
    r_seeds=[1111,123,42,444,54321,5555,789,8888]

    means = {}
    stds = {}
    trj = 15
    for r_s in r_seeds:
        f = open("conv_mul/0_vel2/"+str(r_s)+ ".csv", 'w', newline='')
        writer1 = csv.writer(f)
        writer1.writerow(
            ["lat_cost_mean", "lat_cost_std", "ctrl_cost", "diff_cost", "success", "beta", "log_min_svd","ctrl_cost_std","diff_cost_std","log_max_svd"])
        for beta in betas:
            m_sv1 = []
            ma_sv1=[]
            print("eval: started")

            mp.set_start_method('spawn', force=True)
            p = mp.Pool()
            e2c = E2C()
            e2c.load_state_dict(torch.load("conv_mul/conv"+str(r_s)+"/mod" + str(beta) + ".pth", map_location=device))
            cost, control_cost, dif_cost, succ, m_sv,ma_sv = zip(*p.map(MPC, [e2c] * trj))

            m_sv1.append(m_sv)
            ma_sv1.append(ma_sv)
            p.close()
            p.join()
            print(succ)
            print("mean", cost)

            per = sum(np.array(succ)) / succ.shape[0]
            print(per)
            writer1.writerow(
                [np.mean(np.array(cost)), np.std(np.array(cost)), np.mean(np.array(control_cost)), np.mean(np.array(dif_cost)),
                 per, beta, np.mean(np.array(m_sv1)),np.std(np.array(control_cost)),np.std(np.array(dif_cost)),np.mean(np.array(ma_sv1))])