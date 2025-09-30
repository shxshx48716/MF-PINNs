#### cross
#### k = 0.0001 , v = 1
#### data = 2 * 128 * 128
#### data = 2 * 16 * 16 (test)

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.init as init
from torch.optim.lr_scheduler import StepLR
from itertools import chain
import datetime
import time
from matplotlib import rcParams
import math
torch.pi = math.pi
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

filename_result = "Result_mywork_4.txt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
   torch.cuda.manual_seed(1234)
else:
   torch.manual_seed(1234)


now = datetime.datetime.now()

now_now = now.strftime("%Y-%m-%d %H:%M:%S")
print(f"Result_mywork_4,{now_now}\n")

with open(filename_result, "w") as f:
    f.write(f"Result_mywork_4\n  {now_now}\n")
    f.write(f"Using device: {device}\n")

rcParams.update({'font.size': 16})
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Times New Roman', 'DejaVu Sans', 'Bitstream Vera Sans', 'Arial Unicode MS']


class BeltramiPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.Tanh = nn.Tanh()
        self.Relu = nn.ReLU()

        self.layer1 = nn.Linear(2, 70)
        self.layer2 = nn.Linear(70, 70)
        self.layer3 = nn.Linear(70, 70)
        self.layer4 = nn.Linear(70, 70)
        self.layer5 = nn.Linear(70, 70)
        self.layer6 = nn.Linear(70, 2, bias='False')

        self.layer7 = nn.Linear(2, 70)
        self.layer8 = nn.Linear(70, 70)
        self.layer9 = nn.Linear(70, 70)
        self.layer10 = nn.Linear(70, 70)
        self.layer11 = nn.Linear(70, 70)
        self.layer12 = nn.Linear(70, 2, bias='False')

        self.a = nn.Parameter(torch.FloatTensor([1.0]))
        self.b = nn.Parameter(torch.FloatTensor([1.0]))

        #init.xavier_uniform_(self.layer1.weight)
        #init.xavier_uniform_(self.layer2.weight)
        #init.xavier_uniform_(self.layer3.weight)
        #init.xavier_uniform_(self.layer4.weight)

    def forward(self, xs, ys ,xd, yd):
        S = torch.cat((xs, ys), dim=1).to(device)

        S = self.layer1(S)
        # S = torch.sin(2 * torch.pi * S)
        S = self.Tanh(S * self.a)
        S = self.layer2(S)
        S = self.Tanh(S * self.a)
        S = self.layer3(S)
        S = self.Tanh(S * self.a)
        S = self.layer4(S)
        S = self.Tanh(S * self.a)
        S = self.layer5(S)
        S = self.Tanh(S * self.a)
        S = self.layer6(S)

        psi_s = S[:, 0]
        p_s   = S[:, 1]

        psi_s = psi_s.unsqueeze_(1)
        p_s= p_s.unsqueeze_(1)

        D = torch.cat((xd, yd), dim=1).to(device)

        D = self.layer7(D)
        # D = torch.sin(2 * torch.pi * D)
        D = self.Tanh(D * self.b)
        D = self.layer8(D)
        D = self.Tanh(D * self.b)
        D = self.layer9(D)
        D = self.Tanh(D * self.b)
        D = self.layer10(D)
        D = self.Tanh(D * self.b)
        D = self.layer11(D)
        D = self.Tanh(D * self.b)
        D = self.layer12(D)

        psi_d = D[:, 0]
        p_d   = D[:, 1]

        psi_d = psi_d.unsqueeze_(1)
        p_d = p_d.unsqueeze_(1)

        return psi_s, p_s , psi_d, p_d , self.a, self.b

class PhysicsInformedNN:
    def __init__(self, model, a=1.0, d=1.0, alpha=100, beta=100):
        self.model = model.to(device)
        self.a = a
        self.d = d
        self.alpha = alpha
        self.beta = beta

        self.mse = nn.MSELoss()

    def fun_u_s(x, y):
        return -(torch.sin(torch.pi * x) ** 2) * torch.sin(torch.pi * y) * torch.cos(torch.pi * y)

    def fun_v_s(x, y):
        return torch.sin(torch.pi * x) * torch.cos(torch.pi * x) * (torch.sin(torch.pi * y) ** 2)

    def fun_p_s(x, y):
        return torch.sin(torch.pi * x) * torch.cos(torch.pi * y)

    def fun_u_d(x, y):
        return 0.5 * torch.sin(2 * torch.pi * x) * torch.cos(2 * torch.pi * y)

    def fun_v_d(x, y):
        return  - 0.5 * torch.cos(2 * torch.pi * x) * torch.sin(2 * torch.pi * y)

    def fun_p_d(x, y):
        return torch.sin(torch.pi * x) * torch.cos(torch.pi * y)


    def renew_p(num_boundary,p_s,p_d):
        loss_p = 0.5 * (torch.mean(p_s.detach()) + torch.mean(p_d.detach()))
        p_s = p_s - loss_p
        p_d = p_d - loss_p
        return torch.abs(loss_p), p_s, p_d


    def loss_int_s(x, y, psi, p, flags, nu, K):

        u =      torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)[0]
        v = -1 * torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(x), retain_graph=True, create_graph=True)[0]

        u_x = torch.autograd.grad(u,x,grad_outputs=torch.ones_like(x),retain_graph=True,create_graph=True)[0]
        u_y = torch.autograd.grad(u,y,grad_outputs=torch.ones_like(y),retain_graph=True,create_graph=True)[0]

        v_x = torch.autograd.grad(v,x,grad_outputs=torch.ones_like(x),retain_graph=True,create_graph=True)[0]
        v_y = torch.autograd.grad(v,y,grad_outputs=torch.ones_like(y),retain_graph=True,create_graph=True)[0]

        p_x = torch.autograd.grad(p,x,grad_outputs=torch.ones_like(x),retain_graph=True,create_graph=True)[0]
        p_y = torch.autograd.grad(p,y,grad_outputs=torch.ones_like(y),retain_graph=True,create_graph=True)[0]

        # p_xx = torch.autograd.grad(p_x,x,grad_outputs=torch.ones_like(x),retain_graph=True,create_graph=True)[0]
        # p_yy = torch.autograd.grad(p_y,y, grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)[0]


        u_xx = torch.autograd.grad(u_x, x,grad_outputs=torch.ones_like(x),retain_graph=True,create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y,grad_outputs=torch.ones_like(y),retain_graph=True,create_graph=True)[0]
        u_xy = torch.autograd.grad(u_x, y,grad_outputs=torch.ones_like(y),retain_graph=True, create_graph=True)[0]

        v_xx = torch.autograd.grad(v_x,x,grad_outputs=torch.ones_like(x),retain_graph=True,create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y,y,grad_outputs=torch.ones_like(y),retain_graph=True,create_graph=True)[0]
        v_xy = torch.autograd.grad(v_x,y,grad_outputs=torch.ones_like(y),retain_graph=True, create_graph=True)[0]

        fun_u = PhysicsInformedNN.fun_u_s(x,y)
        fun_v = PhysicsInformedNN.fun_v_s(x,y)
        fun_p = PhysicsInformedNN.fun_p_s(x,y)

        fun_u_x = torch.autograd.grad(fun_u, x, grad_outputs=torch.ones_like(x), retain_graph=True, create_graph=True)[0]
        fun_u_y = torch.autograd.grad(fun_u, y, grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)[0]

        fun_v_x = torch.autograd.grad(fun_v, x, grad_outputs=torch.ones_like(x), retain_graph=True, create_graph=True)[0]
        fun_v_y = torch.autograd.grad(fun_v, y, grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)[0]

        fun_p_x = torch.autograd.grad(fun_p, x, grad_outputs=torch.ones_like(x), retain_graph=True, create_graph=True)[0]
        fun_p_y = torch.autograd.grad(fun_p, y, grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)[0]

        fun_u_xx = torch.autograd.grad(fun_u_x, x, grad_outputs=torch.ones_like(x), retain_graph=True, create_graph=True)[0]
        fun_u_yy = torch.autograd.grad(fun_u_y, y, grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)[0]
        fun_u_xy = torch.autograd.grad(fun_u_x, y, grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)[0]

        fun_v_xx = torch.autograd.grad(fun_v_x, x, grad_outputs=torch.ones_like(x), retain_graph=True, create_graph=True)[0]
        fun_v_yy = torch.autograd.grad(fun_v_y, y, grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)[0]
        fun_v_xy = torch.autograd.grad(fun_v_x, y, grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)[0]

        # f1 =  2 * nu * fun_u_xx - fun_p_x + nu * (fun_u_yy + fun_v_xy)
        # f2 =  nu * (fun_v_xx + fun_u_xy) - fun_p_y + 2 * nu * fun_v_yy

        # f1_x = torch.autograd.grad(f1, x, grad_outputs=torch.ones_like(x), retain_graph=True, create_graph=True)[0]
        # f1_y = torch.autograd.grad(f1, y, grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)[0]

        # f2_x = torch.autograd.grad(f2, x, grad_outputs=torch.ones_like(x), retain_graph=True, create_graph=True)[0]
        # f2_y = torch.autograd.grad(f2, y, grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)[0]

        # psi_xxxx = -1 * torch.autograd.grad(v_xx, x, grad_outputs=torch.ones_like(x), retain_graph=True, create_graph=True)[0]
        # psi_yyyy =      torch.autograd.grad(u_yy, y, grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)[0]
        # psi_xxyy =      torch.autograd.grad(u_xy, x, grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)[0]

        s_int_1 = 0 + 2 * nu * u_xx - p_x + nu * ( u_yy + v_xy ) \
                    - ( 2 * nu * fun_u_xx - fun_p_x + nu * (fun_u_yy + fun_v_xy) )
        s_int_2 = 0 + nu * ( v_xx + u_xy ) - p_y + 2 * nu * v_yy \
                    - ( nu *(fun_v_xx + fun_u_xy) - fun_p_y + 2 * nu * fun_v_yy )

        # s_int_3 = ( nu * ( 2 * psi_xxyy + psi_xxxx + psi_yyyy) - (f1_y - f2_x) ) * (nu**(-0.5))
        # s_int_4 = ( p_xx + p_yy + f1_x + f2_y ) * (nu**0.5)

        condition = flags == 5

        s_int_1 = torch.unsqueeze(s_int_1[condition], dim = 1)
        s_int_2 = torch.unsqueeze(s_int_2[condition], dim = 1)
        # s_int_3 = torch.unsqueeze(s_int_3[condition], dim = 1)
        # s_int_4 = torch.unsqueeze(s_int_4[condition], dim = 1)

        loss_int_s = torch.mean(s_int_1**2) + torch.mean(s_int_2**2) # + torch.mean(s_int_3**2) + torch.mean(s_int_4**2)

        return  u, v, loss_int_s

    def loss_int_d(x, y, psi, p, flagd, nu , K):

        u = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)[0]
        v = -1 * torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(x), retain_graph=True, create_graph=True)[0]

        psi_yy = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(x), retain_graph=True, create_graph=True)[0]
        psi_xx = -1 * torch.autograd.grad(v, x, grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)[0]

        p_x = torch.autograd.grad(p,x,grad_outputs=torch.ones_like(x),retain_graph=True,create_graph=True)[0]
        p_y = torch.autograd.grad(p,y,grad_outputs=torch.ones_like(y),retain_graph=True,create_graph=True)[0]

        # p_xx = torch.autograd.grad(p_x, x, grad_outputs=torch.ones_like(x), retain_graph=True, create_graph=True)[0]
        # p_yy = torch.autograd.grad(p_y, y, grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)[0]

        fun_p = PhysicsInformedNN.fun_p_d(x,y)
        fun_p_x = torch.autograd.grad(fun_p,x,grad_outputs=torch.ones_like(x),retain_graph=True,create_graph=True)[0]
        fun_p_y = torch.autograd.grad(fun_p,y,grad_outputs=torch.ones_like(y),retain_graph=True,create_graph=True)[0]

        f1 = nu * (1/K) * PhysicsInformedNN.fun_u_d(x,y) + fun_p_x
        f2 = nu * (1/K) * PhysicsInformedNN.fun_v_d(x,y) + fun_p_y

        # f1_x = torch.autograd.grad(f1,x,grad_outputs=torch.ones_like(x),retain_graph=True,create_graph=True)[0]
        # f1_y = torch.autograd.grad(f1,y,grad_outputs=torch.ones_like(y),retain_graph=True,create_graph=True)[0]

        # f2_x = torch.autograd.grad(f2,x,grad_outputs=torch.ones_like(x),retain_graph=True,create_graph=True)[0]
        # f2_y = torch.autograd.grad(f2,y,grad_outputs=torch.ones_like(y),retain_graph=True,create_graph=True)[0]

        d_int_1 = nu * (1/K) * u + p_x - f1
        d_int_2 = nu * (1/K) * v + p_y - f2

        # d_int_3 =  ((nu/K) * (psi_xx + psi_yy) - f1_y + f2_x )  * ((nu/K)**(-0.5))
        # d_int_4 =  ( p_xx + p_yy - (f1_x + f2_y) ) * ((nu/K)**(0.5))

        condition = flagd == 10

        d_int_1 = torch.unsqueeze(d_int_1[condition], dim=1)
        d_int_2 = torch.unsqueeze(d_int_2[condition], dim=1)
        # d_int_3 = torch.unsqueeze(d_int_3[condition], dim=1)
        # d_int_4 = torch.unsqueeze(d_int_4[condition], dim=1)

        loss_int_d = torch.mean(d_int_1 ** 2) + torch.mean(d_int_2 ** 2) #  + torch.mean(d_int_3 ** 2) + torch.mean(d_int_4 ** 2)

        return u, v, loss_int_d

    def loss_boundary_d(x, y, u, v, p, flagd):

        condition = flagd == 7
        boundary_error_down = torch.unsqueeze( -v[condition], dim = 1)

        condition = flagd == 6
        boundary_error_left = torch.unsqueeze( -u[condition], dim = 1)

        condition = flagd == 8
        boundary_error_right = torch.unsqueeze( u[condition], dim = 1)

        boundary_error_d = torch.mean(boundary_error_left **2) + torch.mean(boundary_error_down **2) + torch.mean(boundary_error_right **2)

        return boundary_error_d

    def loss_boundary_s(x, y, u, v, p, flags):

        boundary_error_u = u - PhysicsInformedNN.fun_u_s(x,y)
        boundary_error_v = v - PhysicsInformedNN.fun_v_s(x,y)

        condition =  (flags == 1)  |  (flags == 2)  | (flags == 3)
        boundary_error_u = torch.unsqueeze(boundary_error_u[condition], dim=1)
        boundary_error_v = torch.unsqueeze(boundary_error_v[condition], dim=1)

        boundary_error_s = torch.mean(boundary_error_u ** 2) + torch.mean(boundary_error_v ** 2)

        return boundary_error_s

    def loss_boundary_inter(x_s, y_s, u_s, v_s, p_s, x_d, y_d, u_d, v_d, p_d ,flags,flagd,nu,K):

        v_s_y = torch.autograd.grad(v_s, y_s, grad_outputs=torch.ones_like(y_s), retain_graph=True,create_graph=True)[0]
        fun_v_s_y = torch.autograd.grad(PhysicsInformedNN.fun_v_s(x_s,y_s), y_s, grad_outputs=torch.ones_like(y_s), retain_graph=True,create_graph=True)[0]

        v_s_x = torch.autograd.grad(v_s, x_s, grad_outputs=torch.ones_like(x_s), retain_graph=True, create_graph=True)[0]
        u_s_y = torch.autograd.grad(u_s, y_s, grad_outputs=torch.ones_like(y_s), retain_graph=True, create_graph=True)[0]

        fun_v_s_x = torch.autograd.grad(PhysicsInformedNN.fun_v_s(x_s,y_s), x_s, grad_outputs=torch.ones_like(x_s), retain_graph=True, create_graph=True)[0]
        fun_u_s_y = torch.autograd.grad(PhysicsInformedNN.fun_u_s(x_s,y_s), y_s, grad_outputs=torch.ones_like(y_s),retain_graph=True, create_graph=True)[0]

        condition_s = flags == 4

        fun_v_s = torch.unsqueeze(PhysicsInformedNN.fun_v_s(x_s, y_s), dim=1)
        v_s = torch.unsqueeze(v_s[condition_s], dim=1)
        v_s_y = torch.unsqueeze(v_s_y[condition_s], dim=1)
        p_s = torch.unsqueeze(p_s[condition_s], dim=1)
        fun_v_s_y = torch.unsqueeze(fun_v_s_y[condition_s], dim=1)
        fun_p_s = torch.unsqueeze(PhysicsInformedNN.fun_p_s(x_s,y_s)[condition_s], dim=1)
        v_s_x = torch.unsqueeze(v_s_x[condition_s], dim=1)
        u_s_y = torch.unsqueeze(u_s_y[condition_s], dim=1)
        u_s = torch.unsqueeze(u_s[condition_s], dim=1)
        fun_v_s_x = torch.unsqueeze(fun_v_s_x[condition_s], dim=1)
        fun_u_s_y = torch.unsqueeze(fun_u_s_y[condition_s], dim=1)
        fun_u_s = torch.unsqueeze(PhysicsInformedNN.fun_u_s(x_s, y_s)[condition_s], dim=1)

        condition_d = flagd == 9

        fun_v_d = torch.unsqueeze(PhysicsInformedNN.fun_v_d(x_d, y_d), dim=1)
        v_d = torch.unsqueeze(v_d[condition_d], dim=1)
        p_d = torch.unsqueeze(p_d[condition_d], dim=1)
        fun_p_d = torch.unsqueeze(PhysicsInformedNN.fun_p_d(x_d, y_d)[condition_d], dim=1)

        loss_inter_1 =   v_s - v_d - fun_v_s + fun_v_d
        loss_inter_2 =  2 * nu *  v_s_y - p_s + p_d \
                      - 2 * nu * fun_v_s_y + fun_p_s - fun_p_d
        loss_inter_3 =   - 2 * 0.5 * (v_s_x + u_s_y) + 1 * ((1/K)**0.5) * u_s \
                         + 2 * 0.5 * (fun_v_s_x + fun_u_s_y) - ((1/K)**0.5) * fun_u_s

        loss_inter = torch.mean(loss_inter_1**2) + torch.mean(loss_inter_2**2) + torch.mean(loss_inter_3**2)

        return loss_inter

    def L2(x_s, y_s, u_s, v_s, p_s, x_d, y_d, u_d, v_d, p_d):

        L2_u_s = torch.sum((u_s - PhysicsInformedNN.fun_u_s(x_s, y_s)) ** 2) ** 0.5 / torch.sum(PhysicsInformedNN.fun_u_s(x_s, y_s) ** 2) ** 0.5
        L2_v_s = torch.sum((v_s - PhysicsInformedNN.fun_v_s(x_s, y_s)) ** 2) ** 0.5 / torch.sum(PhysicsInformedNN.fun_v_s(x_s, y_s) ** 2) ** 0.5
        L2_p_s = torch.sum((p_s - PhysicsInformedNN.fun_p_s(x_s, y_s)) ** 2) ** 0.5 / torch.sum(PhysicsInformedNN.fun_p_s(x_s, y_s) ** 2) ** 0.5

        L2_u_d = torch.sum((u_d - PhysicsInformedNN.fun_u_d(x_d, y_d)) ** 2) ** 0.5 / torch.sum(PhysicsInformedNN.fun_u_d(x_d, y_d) ** 2) ** 0.5
        L2_v_d = torch.sum((v_d - PhysicsInformedNN.fun_v_d(x_d, y_d)) ** 2) ** 0.5 / torch.sum(PhysicsInformedNN.fun_v_d(x_d, y_d) ** 2) ** 0.5
        L2_p_d = torch.sum((p_d - PhysicsInformedNN.fun_p_d(x_d, y_d)) ** 2) ** 0.5 / torch.sum(PhysicsInformedNN.fun_p_d(x_d, y_d) ** 2) ** 0.5

        L2_u_s_ab = torch.mean((u_s - PhysicsInformedNN.fun_u_s(x_s, y_s)) ** 2) ** 0.5
        L2_v_s_ab = torch.mean((v_s - PhysicsInformedNN.fun_v_s(x_s, y_s)) ** 2) ** 0.5
        L2_p_s_ab = torch.mean((p_s - PhysicsInformedNN.fun_p_s(x_s, y_s)) ** 2) ** 0.5
        L2_u_d_ab = torch.mean((u_d - PhysicsInformedNN.fun_u_d(x_d, y_d)) ** 2) ** 0.5
        L2_v_d_ab = torch.mean((v_d - PhysicsInformedNN.fun_v_d(x_d, y_d)) ** 2) ** 0.5
        L2_p_d_ab = torch.mean((p_d - PhysicsInformedNN.fun_p_d(x_d, y_d)) ** 2) ** 0.5

        return L2_u_s, L2_v_s, L2_p_s, L2_u_d, L2_v_d, L2_p_d , \
            L2_u_s_ab, L2_v_s_ab, L2_p_s_ab, \
            L2_u_d_ab, L2_v_d_ab, L2_p_d_ab

    def data(num_boundary=128):

        xd_boundary = torch.linspace(0, 1, num_boundary, requires_grad=False, device=device)
        yd_boundary = torch.linspace(-1, 0, int(num_boundary), requires_grad=False, device=device)

        xs_boundary = torch.linspace(0, 1, num_boundary  , requires_grad=False, device=device)
        ys_boundary = torch.linspace(0, 1, int(num_boundary), requires_grad=False, device=device)

        flags = torch.zeros(int((num_boundary ** 2)), 1)
        flagd = torch.zeros(int((num_boundary ** 2)), 1)

        xxd, yyd = torch.meshgrid(xd_boundary, yd_boundary, indexing='ij')
        xxs, yys = torch.meshgrid(xs_boundary, ys_boundary, indexing='ij')

        xxd = xxd.flatten()
        yyd = yyd.flatten()
        xxs = xxs.flatten()
        yys = yys.flatten()

        for i in range(flags.shape[0]):
            if   yys[i] == 0:
                 flags[i, 0] = 4
            elif yys[i] == 1:
                 flags[i, 0] = 2
            elif xxs[i] == 0 and yys[i] > 0 :
                 flags[i, 0] = 1
            elif xxs[i] == 1 and yys[i] > 0 :
                 flags[i, 0] = 3
            else:
                 flags[i, 0] = 5

        for i in range(flagd.shape[0]):
            if   yyd[i] == 0:
                 flagd[i, 0] = 9
            elif yyd[i] == -1:
                 flagd[i, 0] = 7
            elif xxd[i] == 0 and yyd[i] < 0:
                 flagd[i, 0] = 6
            elif xxd[i] == 1 and yyd[i] < 0:
                 flagd[i, 0] = 8
            else:
                 flagd[i, 0] = 10

        xxs.unsqueeze_(0)
        yys.unsqueeze_(0)
        xxd.unsqueeze_(0)
        yyd.unsqueeze_(0)

        xs = torch.swapaxes(xxs, 0, 1)
        ys = torch.swapaxes(yys, 0, 1)
        xd = torch.swapaxes(xxd, 0, 1)
        yd = torch.swapaxes(yyd, 0, 1)

        xs.requires_grad_(True)
        ys.requires_grad_(True)
        xd.requires_grad_(True)
        yd.requires_grad_(True)

        return xs, ys, flags, xd, yd, flagd

    def myplot(x_s, y_s, u_s, v_s, p_s, x_d, y_d, u_d, v_d, p_d, num_boundary):

        xs_boundary = np.linspace(0, 1, num_boundary)
        ys_boundary = np.linspace(0, 1, int(num_boundary))
        xd_boundary = np.linspace(0, 1, num_boundary)
        yd_boundary = np.linspace(-1, 0, int(num_boundary))

        xxs, yys = np.meshgrid(xs_boundary, ys_boundary)
        xxd, yyd = np.meshgrid(xd_boundary, yd_boundary)

        true_u_s = PhysicsInformedNN.fun_u_s(x_s, y_s)
        true_v_s = PhysicsInformedNN.fun_v_s(x_s, y_s)
        true_p_s = PhysicsInformedNN.fun_p_s(x_s, y_s)

        error_u_s = torch.abs(u_s - true_u_s)
        error_v_s = torch.abs(v_s - true_v_s)
        error_p_s = torch.abs(p_s - true_p_s)

        u_s = u_s.to('cpu').detach().numpy()
        v_s = v_s.to('cpu').detach().numpy()
        p_s = p_s.to('cpu').detach().numpy()

        error_u_s = error_u_s.to('cpu').detach().numpy()
        error_v_s = error_v_s.to('cpu').detach().numpy()
        error_p_s = error_p_s.to('cpu').detach().numpy()

        true_u_s = true_u_s.to('cpu').detach().numpy()
        true_v_s = true_v_s.to('cpu').detach().numpy()
        true_p_s = true_p_s.to('cpu').detach().numpy()

        uu_s = np.reshape(u_s, (num_boundary, int(num_boundary))).T
        vv_s = np.reshape(v_s, (num_boundary, int(num_boundary))).T
        pp_s = np.reshape(p_s, (num_boundary, int(num_boundary))).T

        error_uu_s = np.reshape(error_u_s, (num_boundary, int(num_boundary))).T
        error_vv_s = np.reshape(error_v_s, (num_boundary, int(num_boundary))).T
        error_pp_s = np.reshape(error_p_s, (num_boundary, int(num_boundary))).T

        true_uu_s = np.reshape(true_u_s, (num_boundary, int(num_boundary))).T
        true_vv_s = np.reshape(true_v_s, (num_boundary, int(num_boundary))).T
        true_pp_s = np.reshape(true_p_s, (num_boundary, int(num_boundary))).T

        true_u_d = PhysicsInformedNN.fun_u_d(x_d, y_d)
        true_v_d = PhysicsInformedNN.fun_v_d(x_d, y_d)
        true_p_d = PhysicsInformedNN.fun_p_d(x_d, y_d)

        error_u_d = torch.abs(u_d - true_u_d)
        error_v_d = torch.abs(v_d - true_v_d)
        error_p_d = torch.abs(p_d - true_p_d)

        u_d = u_d.to('cpu').detach().numpy()
        v_d = v_d.to('cpu').detach().numpy()
        p_d = p_d.to('cpu').detach().numpy()

        error_u_d = error_u_d.to('cpu').detach().numpy()
        error_v_d = error_v_d.to('cpu').detach().numpy()
        error_p_d = error_p_d.to('cpu').detach().numpy()

        true_u_d = true_u_d.to('cpu').detach().numpy()
        true_v_d = true_v_d.to('cpu').detach().numpy()
        true_p_d = true_p_d.to('cpu').detach().numpy()

        uu_d = np.reshape(u_d, (num_boundary, int(num_boundary))).T
        vv_d = np.reshape(v_d, (num_boundary, int(num_boundary))).T
        pp_d = np.reshape(p_d, (num_boundary, int(num_boundary))).T

        error_uu_d = np.reshape(error_u_d, (num_boundary, int(num_boundary))).T
        error_vv_d = np.reshape(error_v_d, (num_boundary, int(num_boundary))).T
        error_pp_d = np.reshape(error_p_d, (num_boundary, int(num_boundary))).T

        true_uu_d = np.reshape(true_u_d, (num_boundary, int(num_boundary))).T
        true_vv_d = np.reshape(true_v_d, (num_boundary, int(num_boundary))).T
        true_pp_d = np.reshape(true_p_d, (num_boundary, int(num_boundary))).T

        plt.figure(figsize=(6, 4.5))
        plt.contourf(xxs, yys, uu_s, levels=100, cmap='jet')
        cbar = plt.colorbar(label='Predict_u_s')
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        plt.title('Predict_u_s')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.tight_layout()
        plt.savefig('u_s_pred_mywork_4.png', dpi=800)
        plt.show()

        plt.figure(figsize=(6, 4.5))
        plt.contourf(xxs, yys, true_uu_s, levels=100, cmap='jet')
        cbar = plt.colorbar(label='True_u_s')
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        plt.title('True_u_s')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.tight_layout()
        plt.savefig('u_s_true_mywork_4.png', dpi=800)
        plt.show()

        plt.figure(figsize=(6, 4.5))
        plt.contourf(xxs, yys, error_uu_s, levels=100, cmap='jet')
        cbar = plt.colorbar(label='Error_u_s')
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        plt.title('Error_u_s')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.tight_layout()
        plt.savefig('u_s_error_mywork_4.png', dpi=800)
        plt.show()

        plt.figure(figsize=(6, 4.5))
        plt.contourf(xxs, yys, vv_s, levels=100, cmap='jet')
        cbar = plt.colorbar(label='Predict_v_s')
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        plt.title('Predict_v_s')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.tight_layout()
        plt.savefig('v_s_pred_mywork_4.png', dpi=800)
        plt.show()

        plt.figure(figsize=(6, 4.5))
        plt.contourf(xxs, yys, true_vv_s, levels=100, cmap='jet')
        cbar = plt.colorbar(label='True_v_s')
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        plt.title('True_v_s')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.tight_layout()
        plt.savefig('v_s_true_mywork_4.png', dpi=800)
        plt.show()

        plt.figure(figsize=(6, 4.5))
        plt.contourf(xxs, yys, error_vv_s, levels=100, cmap='jet')
        cbar = plt.colorbar(label='Error_v_s')
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        plt.title('Error_v_s')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.tight_layout()
        plt.savefig('v_s_error_mywork_4.png', dpi=800)
        plt.show()

        plt.figure(figsize=(6, 4.5))
        plt.contourf(xxs, yys, pp_s, levels=100, cmap='jet')
        cbar = plt.colorbar(label='Predict_p_s')
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        plt.title('Predict_p_s')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.tight_layout()
        plt.savefig('p_s_pred_mywork_4.png', dpi=800)
        plt.show()

        plt.figure(figsize=(6, 4.5))
        plt.contourf(xxs, yys, true_pp_s, levels=100, cmap='jet')
        cbar = plt.colorbar(label='True_p_s')
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        plt.title('True_p_s')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.tight_layout()
        plt.savefig('p_s_true_mywork_4.png', dpi=800)
        plt.show()

        plt.figure(figsize=(6, 4.5))
        plt.contourf(xxs, yys, error_pp_s, levels=100, cmap='jet')
        cbar = plt.colorbar(label='Error_p_s')
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        plt.title('Error_p_s')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.tight_layout()
        plt.savefig('p_s_error_mywork_4.png', dpi=800)
        plt.show()

        plt.figure(figsize=(6, 4.5))
        plt.contourf(xxd, yyd, uu_d, levels=100, cmap='jet')
        cbar = plt.colorbar(label='Predict_u_d')
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        plt.title('Predict_u_d')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([ 0.0, 1.0])
        plt.ylim([-1.0, 0.0])
        plt.tight_layout()
        plt.savefig('u_d_pred_mywork_4.png', dpi=800)
        plt.show()

        plt.figure(figsize=(6, 4.5))
        plt.contourf(xxd, yyd, true_uu_d, levels=100, cmap='jet')
        cbar = plt.colorbar(label='True_u_d')
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        plt.title('True_u_d')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([ 0.0, 1.0])
        plt.ylim([-1.0, 0.0])
        plt.tight_layout()
        plt.savefig('u_d_true_mywork_4.png', dpi=800)
        plt.show()

        plt.figure(figsize=(6, 4.5))
        plt.contourf(xxd, yyd, error_uu_d, levels=100, cmap='jet')
        cbar = plt.colorbar(label='Error_u_d')
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        plt.title('Error_u_d')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([ 0.0, 1.0])
        plt.ylim([-1.0, 0.0])
        plt.tight_layout()
        plt.savefig('u_d_error_mywork_4.png', dpi=800)
        plt.show()

        plt.figure(figsize=(6, 4.5))
        plt.contourf(xxd, yyd, vv_d, levels=100, cmap='jet')
        cbar = plt.colorbar(label='Predict_v_d')
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        plt.title('Predict_v_d')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([ 0.0, 1.0])
        plt.ylim([-1.0, 0.0])
        plt.tight_layout()
        plt.savefig('v_d_pred_mywork_4.png', dpi=800)
        plt.show()

        plt.figure(figsize=(6, 4.5))
        plt.contourf(xxd, yyd, true_vv_d, levels=100, cmap='jet')
        cbar = plt.colorbar(label='True_v_d')
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        plt.title('True_v_d')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([ 0.0, 1.0])
        plt.ylim([-1.0, 0.0])
        plt.tight_layout()
        plt.savefig('v_d_true_mywork_4.png', dpi=800)
        plt.show()

        plt.figure(figsize=(6, 4.5))
        plt.contourf(xxd, yyd, error_vv_d, levels=100, cmap='jet')
        cbar = plt.colorbar(label='Error_v_d')
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        plt.title('Error_v_d')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([ 0.0, 1.0])
        plt.ylim([-1.0, 0.0])
        plt.tight_layout()
        plt.savefig('v_d_error_mywork_4.png', dpi=800)
        plt.show()

        plt.figure(figsize=(6, 4.5))
        plt.contourf(xxd, yyd, pp_d, levels=100, cmap='jet')
        cbar = plt.colorbar(label='Predict_p_d')
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        plt.title('Predict_p_d')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([ 0.0, 1.0])
        plt.ylim([-1.0, 0.0])
        plt.tight_layout()
        plt.savefig('p_d_pred_mywork_4.png', dpi=800)
        plt.show()

        plt.figure(figsize=(6, 4.5))
        plt.contourf(xxd, yyd, true_pp_d, levels=100, cmap='jet')
        cbar = plt.colorbar(label='True_p_d')
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        plt.title('True_p_d')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([ 0.0, 1.0])
        plt.ylim([-1.0, 0.0])
        plt.tight_layout()
        plt.savefig('p_d_true_mywork_4.png', dpi=800)
        plt.show()

        plt.figure(figsize=(6, 4.5))
        plt.contourf(xxd, yyd, error_pp_d, levels=100, cmap='jet')
        cbar = plt.colorbar(label='Error_p_d')
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        plt.title('Error_p_d')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([ 0.0, 1.0])
        plt.ylim([-1.0, 0.0])
        plt.tight_layout()
        plt.savefig('p_d_error_mywork_4.png', dpi=800)
        plt.show()

        return xxs, yys, xxd, yyd, u_s, v_s, p_s, u_d, v_d, p_d, \
            true_u_s, true_v_s, true_p_s, true_u_d, true_v_d, true_p_d, \
            error_u_s, error_v_s, error_p_s, error_u_d, error_v_d, error_p_d

    def my_plot_loss(loss_epochs,loss_int_s_epochs,loss_boundary_s_epochs,\
                     loss_int_d_epochs,loss_boundary_d_epochs,loss_inter,loss_p_epochs,\
                     L2_sum_epochs, L2_u_s_epochs, L2_v_s_epochs, L2_p_s_epochs,\
                     L2_u_d_epochs, L2_v_d_epochs, L2_p_d_epochs, \
                     L2_sum_ab_epochs, L2_u_s_ab_epochs, L2_v_s_ab_epochs, L2_p_s_ab_epochs, \
                     L2_u_d_ab_epochs, L2_v_d_ab_epochs, L2_p_d_ab_epochs, \
                     a_epochs , b_epochs , lr_epochs , adam_num):
        epochs = np.linspace(1,len(loss_epochs),len(loss_epochs))

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        plt.axvline(x = adam_num, linestyle='--', color='gray', linewidth=3)
        plt.plot(epochs, L2_sum_epochs, c='red', label='$L_{2sum}$',linewidth=1)
        plt.plot(epochs, loss_epochs, c='blue', label='$Loss_{sum}$',linewidth=1)
        plt.plot(epochs, loss_int_s_epochs, c='cyan', label='$Loss_{int\_s}$',linewidth=1)
        plt.plot(epochs, loss_boundary_s_epochs, c='aqua', label='$Loss_{bou\_s}$',linewidth=1)
        plt.plot(epochs, loss_int_d_epochs, c='green', label='$Loss_{int\_d}$',linewidth=1)
        plt.plot(epochs, loss_boundary_d_epochs, c='olive', label='$Loss_{bou\_d}$',linewidth=1)
        plt.plot(epochs, loss_inter, c='purple', label='$Loss_{BJS}$',linewidth=1)
        ax.set_yscale('log')
        plt.legend(frameon=True, edgecolor='black', facecolor='white', framealpha=1,fancybox=False)
        plt.xlabel('Epochs')
        plt.ylabel('Metrics')
        plt.savefig('Loss_mywork_4_legend_on.png', dpi=800)
        plt.show()

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        plt.axvline(x = adam_num, linestyle='--', color='gray', linewidth=3)
        plt.plot(epochs, L2_sum_epochs, c='red', label='$L_{2\_sum}$',linewidth=1)
        plt.plot(epochs, loss_epochs, c='blue', label='$Loss_{sum}$',linewidth=1)
        plt.plot(epochs, loss_int_s_epochs, c='cyan', label='$Loss_{int\_s}$',linewidth=1)
        plt.plot(epochs, loss_boundary_s_epochs, c='aqua', label='Loss_{bou\_s}$',linewidth=1)
        plt.plot(epochs, loss_int_d_epochs, c='green', label='$Loss_{int\_d}$',linewidth=1)
        plt.plot(epochs, loss_boundary_d_epochs, c='olive', label='$Loss_{bou\_d}$',linewidth=1)
        plt.plot(epochs, loss_inter, c='purple', label='$Loss_{BJS}$',linewidth=1)
        ax.set_yscale('log')
        # plt.legend(frameon=True, edgecolor='black', facecolor='white', framealpha=1,fancybox=False)
        plt.xlabel('Epochs')
        plt.ylabel('Metrics')
        plt.savefig('Loss_mywork_4_legend_off.png', dpi=800)
        plt.show()

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        plt.axvline(x=adam_num, linestyle='--', color='gray', linewidth=3)
        plt.plot(epochs, L2_sum_epochs, c='red', label='$L_{2sum}$',linewidth=1)
        plt.plot(epochs, L2_u_s_epochs, c='blue', label='$L_{2u_s}$',linewidth=1)
        plt.plot(epochs, L2_v_s_epochs, c='aqua', label='$L_{2v_s}$',linewidth=1)
        plt.plot(epochs, L2_p_s_epochs, c='orange', label='$L_{2p_s}$',linewidth=1)
        plt.plot(epochs, L2_u_d_epochs, c='green', label='$L_{2u_d}$',linewidth=1)
        plt.plot(epochs, L2_v_d_epochs, c='olive', label='$L_{2v_d}$',linewidth=1)
        plt.plot(epochs, L2_p_d_epochs, c='purple', label='$L_{2p_d}$',linewidth=1)
        ax.set_yscale('log')
        plt.legend(frameon=True, edgecolor='black', facecolor='white', framealpha=1,fancybox=False)
        plt.xlabel('Epochs')
        plt.ylabel('L2')
        plt.savefig('L2_mywork_4_legend_on.png', dpi=800)
        plt.show()

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        plt.axvline(x=adam_num, linestyle='--', color='gray', linewidth=3)
        plt.plot(epochs, L2_sum_epochs, c='red', label='$L_{2sum}$',linewidth=1)
        plt.plot(epochs, L2_u_s_epochs, c='blue', label='$L_{2u_s}$',linewidth=1)
        plt.plot(epochs, L2_v_s_epochs, c='aqua', label='$L_{2v_s}$',linewidth=1)
        plt.plot(epochs, L2_p_s_epochs, c='orange', label='$L_{2p_s}$',linewidth=1)
        plt.plot(epochs, L2_u_d_epochs, c='green', label='$L_{2u_d}$',linewidth=1)
        plt.plot(epochs, L2_v_d_epochs, c='olive', label='$L_{2v_d}$',linewidth=1)
        plt.plot(epochs, L2_p_d_epochs, c='purple', label='$L_{2p_d}$',linewidth=1)
        ax.set_yscale('log')
        # plt.legend(frameon=True, edgecolor='black', facecolor='white', framealpha=1,fancybox=False)
        plt.xlabel('Epochs')
        plt.ylabel('L2')
        plt.savefig('L2_mywork_4_legend_off.png', dpi=800)
        plt.show()

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        plt.axvline(x=adam_num, linestyle='--', color='gray', linewidth=3)
        plt.plot(epochs, L2_sum_ab_epochs, c='red', label='$L_{2sum \_ ab}$',linewidth=1)
        plt.plot(epochs, L2_u_s_ab_epochs, c='blue', label='$L_{2u_s \_ ab}$',linewidth=1)
        plt.plot(epochs, L2_v_s_ab_epochs, c='aqua', label='$L_{2v_s \_ ab}$',linewidth=1)
        plt.plot(epochs, L2_p_s_ab_epochs, c='orange', label='$L_{2p_s \_ ab}$',linewidth=1)
        plt.plot(epochs, L2_u_d_ab_epochs, c='green', label='$L_{2u_d \_ ab}$',linewidth=1)
        plt.plot(epochs, L2_v_d_ab_epochs, c='olive', label='$L_{2v_d \_ ab}$',linewidth=1)
        plt.plot(epochs, L2_p_d_ab_epochs, c='purple', label='$L_{2p_d \_ ab}$',linewidth=1)
        ax.set_yscale('log')
        plt.legend(frameon=True, edgecolor='black', facecolor='white', framealpha=1,fancybox=False)
        plt.xlabel('Epochs')
        plt.ylabel('L2_ab')
        plt.savefig('L2_ab_mywork_4_legend_on.png', dpi=800)
        plt.show()

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        plt.axvline(x=adam_num, linestyle='--', color='gray', linewidth=3)
        plt.plot(epochs, L2_sum_ab_epochs, c='red', label='$L_{2sum \_ ab}$',linewidth=1)
        plt.plot(epochs, L2_u_s_ab_epochs, c='blue', label='$L_{2u_s \_ ab}$',linewidth=1)
        plt.plot(epochs, L2_v_s_ab_epochs, c='aqua', label='$L_{2v_s \_ ab}$',linewidth=1)
        plt.plot(epochs, L2_p_s_ab_epochs, c='orange', label='$L_{2p_s \_ ab}$',linewidth=1)
        plt.plot(epochs, L2_u_d_ab_epochs, c='green', label='$L_{2u_d \_ ab}$',linewidth=1)
        plt.plot(epochs, L2_v_d_ab_epochs, c='olive', label='$L_{2v_d \_ ab}$',linewidth=1)
        plt.plot(epochs, L2_p_d_ab_epochs, c='purple', label='$L_{2p_d \;_ ab}$',linewidth=1)
        ax.set_yscale('log')
        # plt.legend(frameon=True, edgecolor='black', facecolor='white', framealpha=1,fancybox=False)
        plt.xlabel('Epochs')
        plt.ylabel('L2_ab')
        plt.savefig('L2_ab_mywork_4_legend_off.png', dpi=800)
        plt.show()

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        plt.axvline(x=adam_num, linestyle='--', color='gray', linewidth=3)
        plt.plot(epochs, a_epochs , label='$a_{epochs}$', c='red',  marker='X', markersize=8,linestyle='None',markerfacecolor='None', markeredgecolor='red')
        plt.plot(epochs, b_epochs , label='$b_{epochs}$', c='blue', marker='o', markersize=8,linestyle='None',markerfacecolor='None', markeredgecolor='blue')
        plt.legend(frameon=True, edgecolor='black', facecolor='white', framealpha=1, fancybox=False)
        plt.xlabel('Epochs')
        plt.ylabel('a,b')
        plt.savefig('a,b_mywork_4.png', dpi=800)
        plt.show()

        fig = plt.figure(figsize=(10, 8))

        ax = fig.add_subplot(111)
        plt.axvline(x=adam_num, linestyle='--', color='gray', linewidth=3)
        plt.plot(epochs, lr_epochs, label='$learning \: rate_{epochs}$', marker='^', markersize=8,linestyle='None', c='red',markerfacecolor='None', markeredgecolor='red')
        ax.set_yscale('log')
        plt.legend(frameon=True, edgecolor='black', facecolor='white', framealpha=1,fancybox=False)
        plt.xlabel('Epochs')
        plt.ylabel('lr')
        plt.savefig('lr_mywork_4.png', dpi=800)
        plt.show()

        return None

def train():
    epochs_adam = 7000
    epochs_LBFGS = 3000
    num_boundary = 128
    nu = 1
    K = 0.0001

    model = BeltramiPINN()
    pinn = PhysicsInformedNN(model)

    x_s, y_s, flags, x_d, y_d, flagd = PhysicsInformedNN.data()

    a_epochs = []
    b_epochs = []
    lr_epochs = []

    loss_epochs = []  
    loss_int_s_epochs = []
    loss_boundary_s_epochs = []
    loss_int_d_epochs = []
    loss_boundary_d_epochs = []
    loss_boundary_inter_epochs=[]
    loss_p_epochs=[]

    L2_sum_epochs = []
    L2_u_s_epochs = []
    L2_v_s_epochs = []
    L2_p_s_epochs = []
    L2_u_d_epochs = []
    L2_v_d_epochs = []
    L2_p_d_epochs = []

    L2_sum_ab_epochs = []
    L2_u_s_ab_epochs = []
    L2_v_s_ab_epochs = []
    L2_p_s_ab_epochs = []
    L2_u_d_ab_epochs = []
    L2_v_d_ab_epochs = []
    L2_p_d_ab_epochs = []

    All_start = time.time()  

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                               patience=100, verbose=True, threshold=0.0001, threshold_mode='rel',
                                               cooldown=100, min_lr=0, eps=1e-08)
    for epoch in range(1, 1 + epochs_adam):
        start = time.time()
        optimizer.zero_grad()
        psi_s, p_s, psi_d, p_d, a , b =  model(x_s,y_s,x_d,y_d)
        u_s,v_s,loss_int_s =  PhysicsInformedNN.loss_int_s(x_s, y_s, psi_s, p_s, flags,nu,K)
        u_d,v_d,loss_int_d =  PhysicsInformedNN.loss_int_d(x_d, y_d, psi_d, p_d, flagd,nu,K)
        loss_boundary_s =  PhysicsInformedNN.loss_boundary_s(x_s, y_s, u_s, v_s, p_s, flags)
        loss_boundary_d =  PhysicsInformedNN.loss_boundary_d(x_d, y_d, u_d, v_d, p_d, flagd)
        loss_boundary_inter = PhysicsInformedNN.loss_boundary_inter(x_s, y_s, u_s, v_s, p_s, x_d, y_d, u_d, v_d, p_d ,flags,flagd,nu,K)

        loss_s = loss_int_s  + loss_boundary_s + loss_boundary_inter
        loss_d = loss_int_d  + loss_boundary_d + loss_boundary_inter

        if int(epoch/100)%2 == 0:
            for param in chain(model.layer1.parameters(),model.layer2.parameters(),\
                    model.layer3.parameters(),model.layer4.parameters(),\
                    model.layer5.parameters(),model.layer6.parameters()):
                param.requires_grad = True
            for param in chain(model.layer7.parameters(),model.layer8.parameters(),\
                    model.layer9.parameters(),model.layer10.parameters(),\
                    model.layer11.parameters(),model.layer12.parameters()):
                param.requires_grad = False
            loss_s.backward()
            loss = loss_s
            optimizer.step()
            scheduler.step(loss_s)
        else:
            for param in chain(model.layer1.parameters(),model.layer2.parameters(),\
                    model.layer3.parameters(),model.layer4.parameters(),\
                    model.layer5.parameters(),model.layer6.parameters()):
                param.requires_grad = False
            for param in chain(model.layer7.parameters(),model.layer8.parameters(),\
                    model.layer9.parameters(),model.layer10.parameters(),\
                    model.layer11.parameters(),model.layer12.parameters()):
                param.requires_grad = True
            loss_d.backward()
            loss = loss_d
            optimizer.step()
            scheduler.step(loss_d)

        loss_p,p_s, p_d = PhysicsInformedNN.renew_p(num_boundary,p_s,p_d)
        L2_u_s, L2_v_s, L2_p_s, L2_u_d, L2_v_d, L2_p_d , \
        L2_u_s_ab,L2_v_s_ab, L2_p_s_ab, L2_u_d_ab, L2_v_d_ab, L2_p_d_ab \
        = PhysicsInformedNN.L2(x_s, y_s, u_s, v_s, p_s, x_d, y_d, u_d, v_d, p_d)
        L2_sum =  L2_u_s + L2_v_s + L2_p_s + L2_u_d + L2_v_d + L2_p_d
        L2_sum_ab = L2_u_s_ab + L2_v_s_ab + L2_p_s_ab + L2_u_d_ab + L2_v_d_ab + L2_p_d_ab
        end = time.time()

        a_epochs.append(a.item())
        b_epochs.append(b.item())
        lr_epochs.append(optimizer.param_groups[0]['lr'])
        loss_epochs.append(loss.item())
        loss_int_s_epochs.append(loss_int_s.item())
        loss_boundary_s_epochs.append(loss_boundary_s.item())
        loss_int_d_epochs.append(loss_int_d.item())
        loss_boundary_d_epochs.append(loss_boundary_d.item())
        loss_boundary_inter_epochs.append(loss_boundary_inter.item())
        loss_p_epochs.append(loss_p.item())

        L2_sum_epochs.append(L2_sum.item())

        L2_u_s_epochs.append(L2_u_s.item())
        L2_v_s_epochs.append(L2_v_s.item())
        L2_p_s_epochs.append(L2_p_s.item())

        L2_u_d_epochs.append(L2_u_d.item())
        L2_v_d_epochs.append(L2_v_d.item())
        L2_p_d_epochs.append(L2_p_d.item())

        L2_sum_ab_epochs.append(L2_sum_ab.item())

        L2_u_s_ab_epochs.append(L2_u_s_ab.item())
        L2_v_s_ab_epochs.append(L2_v_s_ab.item())
        L2_p_s_ab_epochs.append(L2_p_s_ab.item())

        L2_u_d_ab_epochs.append(L2_u_d_ab.item())
        L2_v_d_ab_epochs.append(L2_v_d_ab.item())
        L2_p_d_ab_epochs.append(L2_p_d_ab.item())

        if epoch % 1 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item()} | "
                  f"Loss_int_s: {loss_int_s.item()} |  Loss_b_s: {loss_boundary_s.item()} | Loss_int_d: {loss_int_d.item()} |"
                  f"Loss_b_d: {loss_boundary_d.item()} | Loss_boundary_inter: {loss_boundary_inter.item()} | Loss_p:{loss_p.item()} | "
                  f"| Time: {end - start}s | "
                  f"L2_sum: {L2_sum.item()} | L2_u_s: {L2_u_s.item()} | L2_v_s: {L2_v_s.item()} | L2_p_s: {L2_p_s.item()} | "
                  f"L2_u_d: {L2_u_d.item()} | L2_v_d: {L2_v_d.item()} | L2_p_d: {L2_p_d.item()} | "
                  f"L2_sum_ab: {L2_sum_ab.item()} | L2_u_s_ab: {L2_u_s_ab.item()} | L2_v_s_ab: {L2_v_s_ab.item()} | L2_p_s_ab: {L2_p_s_ab.item()} | "
                  f"L2_u_d_ab: {L2_u_d_ab.item()} | L2_v_d_ab: {L2_v_d_ab.item()} | L2_p_d_ab: {L2_p_d_ab.item()} | "
                  f" a : {a.item()} | b : {b.item()} | Learning rate : {optimizer.param_groups[0]['lr']}"
                  f" Optimizer:Adam"
                  f"\n")
            with open(filename_result, "a") as f:
                f.write(f"Epoch {epoch} | Loss: {loss.item()} | "
                  f"Loss_int_s: {loss_int_s.item()} |  Loss_b_s: {loss_boundary_s.item()} | Loss_int_d: {loss_int_d.item()} |"
                  f"Loss_b_d: {loss_boundary_d.item()} | Loss_boundary_inter: {loss_boundary_inter.item()} | Loss_p:{loss_p.item()} | "
                  f"| Time: {end - start}s | "
                  f"L2_sum: {L2_sum.item()} | L2_u_s: {L2_u_s.item()} | L2_v_s: {L2_v_s.item()} | L2_p_s: {L2_p_s.item()} | "
                  f"L2_u_d: {L2_u_d.item()} | L2_v_d: {L2_v_d.item()} | L2_p_d: {L2_p_d.item()} | "
                  f"L2_sum_ab: {L2_sum_ab.item()} | L2_u_s_ab: {L2_u_s_ab.item()} | L2_v_s_ab: {L2_v_s_ab.item()} | L2_p_s_ab: {L2_p_s_ab.item()} |"
                  f"L2_u_d_ab: {L2_u_d_ab.item()} | L2_v_d_ab: {L2_v_d_ab.item()} | L2_p_d_ab: {L2_p_d_ab.item()} |"
                  f" a : {a.item()} | b : {b.item()} | Learning rate : {optimizer.param_groups[0]['lr']}"
                  f" Optimizer:Adam"
                  f"\n")

    optimizer = torch.optim.LBFGS(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=0.1, max_iter=20, max_eval=20*1.25, tolerance_grad=1e-64, tolerance_change=1e-64, history_size=100, line_search_fn=None)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                               patience=10, verbose=True, threshold=0.001, threshold_mode='rel',
                                               cooldown=100, min_lr=0, eps=1e-08)
    def closure():
        optimizer.zero_grad()
        psi_s, p_s, psi_d, p_d, a , b =  model(x_s,y_s,x_d,y_d)
        u_s, v_s, loss_int_s = PhysicsInformedNN.loss_int_s(x_s, y_s, psi_s, p_s, flags, nu,K)
        u_d, v_d, loss_int_d = PhysicsInformedNN.loss_int_d(x_d, y_d, psi_d, p_d, flagd, nu,K)
        loss_boundary_s =  PhysicsInformedNN.loss_boundary_s(x_s, y_s, u_s, v_s, p_s, flags)
        loss_boundary_d =  PhysicsInformedNN.loss_boundary_d(x_d, y_d, u_d, v_d, p_d, flagd)
        loss_boundary_inter = PhysicsInformedNN.loss_boundary_inter(x_s, y_s, u_s, v_s, p_s, x_d, y_d, u_d, v_d, p_d ,flags,flagd,nu,K)
        loss = loss_int_s + loss_int_d + loss_boundary_s + loss_boundary_d + loss_boundary_inter
        loss.backward()
        loss_p, p_s, p_d = PhysicsInformedNN.renew_p(num_boundary,p_s,p_d)
        return loss

    for epoch in range(1 ,epochs_LBFGS + 1):
        for param in chain(model.layer1.parameters(), model.layer2.parameters(), \
                           model.layer3.parameters(), model.layer4.parameters(), \
                           model.layer5.parameters(), model.layer6.parameters(),\
                           model.layer7.parameters(), model.layer8.parameters(), \
                           model.layer9.parameters(), model.layer10.parameters(), \
                           model.layer11.parameters(), model.layer12.parameters()):
            param.requires_grad = True
        start = time.time()
        psi_s, p_s, psi_d, p_d, a , b =  model(x_s,y_s,x_d,y_d)
        u_s, v_s, loss_int_s = PhysicsInformedNN.loss_int_s(x_s, y_s, psi_s, p_s, flags, nu,K)
        u_d, v_d, loss_int_d = PhysicsInformedNN.loss_int_d(x_d, y_d, psi_d, p_d, flagd, nu,K)
        loss_boundary_s = PhysicsInformedNN.loss_boundary_s(x_s, y_s, u_s, v_s, p_s, flags)
        loss_boundary_d = PhysicsInformedNN.loss_boundary_d(x_d, y_d, u_d, v_d, p_d, flagd)
        loss_boundary_inter = PhysicsInformedNN.loss_boundary_inter(x_s, y_s, u_s, v_s, p_s, x_d, y_d, u_d, v_d, p_d ,flags,flagd,nu,K)
        loss =  loss_int_s + loss_int_d + loss_boundary_s  +  loss_boundary_d  + loss_boundary_inter
        loss_p, p_s, p_d = PhysicsInformedNN.renew_p(num_boundary,p_s,p_d)
        L2_u_s, L2_v_s, L2_p_s, L2_u_d, L2_v_d, L2_p_d , \
        L2_u_s_ab, L2_v_s_ab, L2_p_s_ab, L2_u_d_ab, L2_v_d_ab, L2_p_d_ab \
        = PhysicsInformedNN.L2(x_s, y_s, u_s, v_s, p_s, x_d, y_d, u_d, v_d, p_d)
        L2_sum =  L2_u_s + L2_v_s + L2_p_s + L2_u_d + L2_v_d + L2_p_d
        L2_sum_ab = L2_u_s_ab + L2_v_s_ab + L2_p_s_ab + L2_u_d_ab + L2_v_d_ab + L2_p_d_ab
        loss = optimizer.step(closure)
        scheduler.step(loss)
        end = time.time()

        if epoch % 1 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item()} | "
                 f"Loss_int_s: {loss_int_s.item()} |  Loss_b_s: {loss_boundary_s.item()} | Loss_int_d: {loss_int_d.item()} |"
                  f"Loss_b_d: {loss_boundary_d.item()} | Loss_boundary_inter: {loss_boundary_inter.item()} | Loss_p:{loss_p.item()} | "
                  f"| Time: {end - start}s | "
                  f"L2_sum: {L2_sum.item()} | L2_u_s: {L2_u_s.item()} | L2_v_s: {L2_v_s.item()} | L2_p_s: {L2_p_s.item()} |"
                  f"L2_u_d: {L2_u_d.item()} | L2_v_d: {L2_v_d.item()} | L2_p_d: {L2_p_d.item()} | "
                  f"L2_sum_ab: {L2_sum_ab.item()} | L2_u_s_ab: {L2_u_s_ab.item()} | L2_v_s_ab: {L2_v_s_ab.item()} | L2_p_s_ab: {L2_p_s_ab.item()} |"
                  f"L2_u_d_ab: {L2_u_d_ab.item()} | L2_v_d_ab: {L2_v_d_ab.item()} | L2_p_d_ab: {L2_p_d_ab.item()} |  "
                  f" a : {a.item()} | b : {b.item()} | Learning rate : {optimizer.param_groups[0]['lr']} |"
                  f" Optimizer:LBFGS"
                  f"\n")
            with open(filename_result, "a") as f:
                f.write(f"Epoch {epoch} | Loss: {loss.item()} | "
                  f"Loss_int_s: {loss_int_s.item()} |  Loss_b_s: {loss_boundary_s.item()} | Loss_int_d: {loss_int_d.item()} |"
                  f"Loss_b_d: {loss_boundary_d.item()} | Loss_boundary_inter: {loss_boundary_inter.item()} | Loss_p:{loss_p.item()} | "
                  f"| Time: {end - start}s | "
                  f"L2_sum: {L2_sum.item()} | L2_u_s: {L2_u_s.item()} | L2_v_s: {L2_v_s.item()} | L2_p_s: {L2_p_s.item()} |"
                  f"L2_u_d: {L2_u_d.item()} | L2_v_d: {L2_v_d.item()} | L2_p_d: {L2_p_d.item()} | "
                  f"L2_sum_ab: {L2_sum_ab.item()} | L2_u_s_ab: {L2_u_s_ab.item()} | L2_v_s_ab: {L2_v_s_ab.item()} | L2_p_s_ab: {L2_p_s_ab.item()} | "
                  f"L2_u_d_ab: {L2_u_d_ab.item()} | L2_v_d_ab: {L2_v_d_ab.item()} | L2_p_d_ab: {L2_p_d_ab.item()} | "
                  f" a : {a.item()} | b : {b.item()} | Learning rate : {optimizer.param_groups[0]['lr']} |"
                  f" Optimizer:LBFGS"
                  f"\n")

        a_epochs.append(a.item())
        b_epochs.append(b.item())
        lr_epochs.append(optimizer.param_groups[0]['lr'])
        loss_epochs.append(loss.item())
        loss_int_s_epochs.append(loss_int_s.item())
        loss_boundary_s_epochs.append(loss_boundary_s.item())
        loss_int_d_epochs.append(loss_int_d.item())
        loss_boundary_d_epochs.append(loss_boundary_d.item())
        loss_boundary_inter_epochs.append(loss_boundary_inter.item())
        L2_sum_epochs.append(L2_sum.item())
        loss_p_epochs.append(loss_p.item())

        L2_u_s_epochs.append(L2_u_s.item())
        L2_v_s_epochs.append(L2_v_s.item())
        L2_p_s_epochs.append(L2_p_s.item())

        L2_u_d_epochs.append(L2_u_d.item())
        L2_v_d_epochs.append(L2_v_d.item())
        L2_p_d_epochs.append(L2_p_d.item())

        L2_sum_ab_epochs.append(L2_sum_ab.item())

        L2_u_s_ab_epochs.append(L2_u_s_ab.item())
        L2_v_s_ab_epochs.append(L2_v_s_ab.item())
        L2_p_s_ab_epochs.append(L2_p_s_ab.item())

        L2_u_d_ab_epochs.append(L2_u_d_ab.item())
        L2_v_d_ab_epochs.append(L2_v_d_ab.item())
        L2_p_d_ab_epochs.append(L2_p_d_ab.item())

    PhysicsInformedNN.my_plot_loss(loss_epochs, loss_int_s_epochs, loss_boundary_s_epochs, \
                 loss_int_d_epochs, loss_boundary_d_epochs, loss_boundary_inter_epochs, loss_p_epochs,\
                 L2_sum_epochs, L2_u_s_epochs, L2_v_s_epochs, L2_p_s_epochs, \
                 L2_u_d_epochs, L2_v_d_epochs, L2_p_d_epochs,\
                 L2_sum_ab_epochs , L2_u_s_ab_epochs , L2_v_s_ab_epochs, L2_p_s_ab_epochs, \
                 L2_u_d_ab_epochs, L2_v_d_ab_epochs, L2_p_d_ab_epochs , \
                 a_epochs, b_epochs , lr_epochs , epochs_adam)

    All_end = time.time()
    print(f"All of time : {All_end - All_start}s")
    with open(filename_result, "a") as f:
        f.write(f"All of time : {All_end - All_start}s\n")

    xxs, yys, xxd, yyd, u_s, v_s, p_s, u_d, v_d, p_d, \
    true_u_s, true_v_s, true_p_s, true_u_d, true_v_d, true_p_d,\
    error_u_s, error_v_s, error_p_s, error_u_d, error_v_d, error_p_d \
    = PhysicsInformedNN.myplot(x_s, y_s, u_s, v_s, p_s, x_d, y_d, u_d, v_d, p_d, num_boundary)


    torch.save(model, 'model_mywork_4.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_mywork_4 = torch.load('model_mywork_4.pth', map_location=device)
    model_mywork_4.to(device)

    filename1 = "data_mywork_4.txt"

    with open(filename1, "a") as f:

        f.write(f"write xxs\n")
        for row in xxs:
            f.write(' '.join(f"{item:.16f}" for item in row) + '\n')

        f.write(f"write xxd\n")
        for row in xxd:
            f.write(' '.join(f"{item:.16f}" for item in row) + '\n')

        f.write(f"write yys\n")
        for row in yys:
            f.write(' '.join(f"{item:.16f}" for item in row) + '\n')

        f.write(f"write yyd\n")
        for row in yyd:
            f.write(' '.join(f"{item:.16f}" for item in row) + '\n')

        f.write(f"write pre_u_s\n")
        for row in u_s:
            f.write(' '.join(f"{item:.16f}" for item in row) + '\n')

        f.write(f"write pre_v_s\n")
        for row in v_s:
            f.write(' '.join(f"{item:.16f}" for item in row) + '\n')

        f.write(f"write pre_u_s\n")
        for row in u_s:
            f.write(' '.join(f"{item:.16f}" for item in row) + '\n')

        f.write(f"write pre_u_d\n")
        for row in u_d:
            f.write(' '.join(f"{item:.16f}" for item in row) + '\n')

        f.write(f"write pre_v_d\n")
        for row in v_d:
            f.write(' '.join(f"{item:.16f}" for item in row) + '\n')

        f.write(f"write pre_p_d\n")
        for row in p_d:
            f.write(' '.join(f"{item:.16f}" for item in row) + '\n')

        f.write(f"write true_u_s\n")
        for row in true_u_s:
            f.write(' '.join(f"{item:.16f}" for item in row) + '\n')

        f.write(f"write true_v_s\n")
        for row in true_v_s:
            f.write(' '.join(f"{item:.16f}" for item in row) + '\n')

        f.write(f"write true_p_s\n")
        for row in true_p_s:
            f.write(' '.join(f"{item:.16f}" for item in row) + '\n')

        f.write(f"write true_u_d\n")
        for row in true_u_d:
            f.write(' '.join(f"{item:.16f}" for item in row) + '\n')

        f.write(f"write true_v_d\n")
        for row in true_v_d:
            f.write(' '.join(f"{item:.16f}" for item in row) + '\n')

        f.write(f"write true_p_d\n")
        for row in true_p_d:
            f.write(' '.join(f"{item:.16f}" for item in row) + '\n')

        f.write(f"write error_u_s\n")
        for row in error_u_s:
            f.write(' '.join(f"{item:.16f}" for item in row) + '\n')

        f.write(f"write error_v_s\n")
        for row in error_v_s:
            f.write(' '.join(f"{item:.16f}" for item in row) + '\n')

        f.write(f"write error_p_s\n")
        for row in error_p_s:
            f.write(' '.join(f"{item:.16f}" for item in row) + '\n')

        f.write(f"write error_u_d\n")
        for row in error_u_d:
            f.write(' '.join(f"{item:.16f}" for item in row) + '\n')

        f.write(f"write error_v_d\n")
        for row in error_v_d:
            f.write(' '.join(f"{item:.16f}" for item in row) + '\n')

        f.write(f"write error_p_d\n")
        for row in error_p_d:
            f.write(' '.join(f"{item:.16f}" for item in row) + '\n')

        f.write(f"L2_sum: {L2_sum.item()} | L2_u_s: {L2_u_s.item()} | L2_v_s: {L2_v_s.item()} | L2_p_s: {L2_p_s.item()} | "
                f"L2_u_d: {L2_u_d.item()} | L2_v_d: {L2_v_d.item()} | L2_p_d: {L2_p_d.item()} | \n"
                f"L2_sum_ab: {L2_sum_ab.item()} | L2_u_s_ab: {L2_u_s_ab.item()} | L2_v_s_ab: {L2_v_s_ab.item()} | L2_p_s_ab: {L2_p_s_ab.item()} \n"
                f"L2_u_d_ab: {L2_u_d_ab.item()} | L2_v_d_ab: {L2_v_d_ab.item()} | L2_p_d_ab: {L2_p_d_ab.item()} | "
                f" a : {a.item()} | b : {b.item()}"
                )


if __name__ == "__main__":
    train()