import torch
import torch.nn as nn


# 假设下面这个是已经训练好的deeponet模型
class TrainedDeeponet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t, x_t):
        return t * x_t


class RKnet(nn.Module):
    def __init__(self, m=4, h=0.01):
        super().__init__()
        # 指定参数为“训练参数”：用nn.Parameter()把需要的tensor围起来就行
        self.lbd = nn.Parameter(torch.ones(m))  # lambda, shape:(m,)
        self.alpha = nn.Parameter(torch.ones(m - 1))  # alpha, shape:(m-1,)
        self.Beta = nn.Parameter(torch.ones([m - 1, m - 1]))  # Beta, shape:(m-1, m-1)

        self.trained_deeponet = TrainedDeeponet()
        self.m = m
        self.h = h

    def forward(self, x):
        K_list = []
        K_1 = self.h * self.trained_deeponet(x[:, 0:1], x[:, 1:2])  # K1 = h * f(t,xt) 表达式特殊，单独写出来
        K_list.append(K_1)
        # 下面循环是计算 K2,K3,.... 每个循环计算一个Ki并存在K_list中
        for i in range(self.m - 1):
            t_temp = x[:, 0:1] + self.alpha[i] * self.h  # f的第一个输入 = t + alpha_i * h
            x_temp = x[:, 1:2] + sum(self.Beta[i][j] * K for j, K in enumerate(K_list))  # f的第二个输入 = xt + Σ(beta_ij * Kj)
            K_temp = self.h * self.trained_deeponet(t_temp, x_temp)  # Ki = h * f(f的第一个输入,f的第二个输入)
            K_list.append(K_temp)
        output = x[:, 1:2] + sum(self.lbd[i] * K for i, K in enumerate(K_list))  # x_t+h = xt + Σ(lambda_i * Ki)
        return output


if __name__ == '__main__':
    mdoel = RKnet()
    # input shape (n,2)
    # output shape (n,1)
    input = torch.tensor([[1, 1],
                          [1, 2],
                          [1, 3]])
    output = mdoel(input)
    pass
