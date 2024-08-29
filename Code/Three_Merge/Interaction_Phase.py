import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from PR_FSCN_Phase import ExtractNetwork

class Self_dimMultiHead_Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='scaled_dot_product', dropout=0):
        super(Self_dimMultiHead_Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_kx = nn.Parameter(torch.FloatTensor(1, embed_dim, embed_dim))
        self.w_qx = nn.Parameter(torch.FloatTensor(1, embed_dim, embed_dim))
        self.proj = nn.Linear(embed_dim, out_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim*2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters() # 对参数进行预处理
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.embed_dim)
        self.w_kx.data.uniform_(-stdv, stdv).renorm_(2, 1, 1e-5).mul_(1e5)
        self.w_qx.data.uniform_(-stdv, stdv).renorm_(2, 1, 1e-5).mul_(1e5)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, k, q, flag):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
        # 后面是对原始特征进行加权求和
        kx_source = k.view(mb_size, k_len, self.n_head, self.hidden_dim).permute(0, 2, 1, 3).contiguous()
        # 类似于获取 特征和查询向量的隐藏表示
        kx = k.view(1, -1, self.embed_dim)

        kx = torch.sigmoid(torch.bmm(kx, self.w_kx).view(-1, k_len, self.embed_dim))
        if flag == True:
            qx = q
        else:
            qx = q.view(1, -1, self.embed_dim)  # 类似于全连接
            qx = torch.sigmoid(torch.bmm(qx, self.w_qx).view(-1, q_len, self.embed_dim))
        if self.score_function == 'scaled_dot_product':
            k_nhead = kx.view(mb_size, k_len, self.n_head, self.hidden_dim).permute(0, 2, 1, 3).contiguous()# [256,2,21,128]
            kt = k_nhead.permute(0, 1, 3, 2).contiguous()# [256,2,128,21]
            qx = qx.view(mb_size, q_len, self.n_head, self.hidden_dim).permute(0, 2, 1, 3).contiguous() # [256,2,1,128]
            qkt = torch.matmul(qx, kt) # [256, 2, 1, 21]
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            score = F.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=-1)
        output = torch.matmul(score, kx_source)  # [256, 2, 1, 128]
        output = output.permute(0, 2, 1, 3).contiguous().view(mb_size, q_len, self.n_head * self.hidden_dim)
        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        return output

class Self_dimMultiHead_Attention1(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='scaled_dot_product', dropout=0):
        super(Self_dimMultiHead_Attention1, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_kx = nn.ParameterList([torch.FloatTensor(1, embed_dim, hidden_dim) for _ in range(n_head)])
        self.w_qx = nn.ParameterList([torch.FloatTensor(1, embed_dim, hidden_dim) for _ in range(n_head)])
        self.proj = nn.Linear(embed_dim, out_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim*2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters() # 对参数进行预处理
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        for i in range(self.n_head):
            self.w_kx[i].data.uniform_(-stdv, stdv).renorm_(2, 1, 1e-5).mul_(1e5)
            self.w_qx[i].data.uniform_(-stdv, stdv).renorm_(2, 1, 1e-5).mul_(1e5)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, k, q, flag):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
        out_list = []
        for i in range(self.n_head):
            # 类似于获取 特征和查询向量的隐藏表示
            kx = k.view(1, -1, self.embed_dim)
            kx = torch.sigmoid(torch.bmm(kx, self.w_kx[i]).view(-1, k_len, self.hidden_dim))
            if flag == True:
                qx = q
            else:
                qx = q.view(1, -1, self.embed_dim)  # 类似于全连接
                qx = torch.sigmoid(torch.bmm(qx, self.w_qx[i]).view(-1, q_len, self.hidden_dim))
            if self.score_function == 'scaled_dot_product':
                kt = kx.permute(0, 2, 1).contiguous()# [256,128,16]
                qkt = torch.matmul(qx, kt) # [256, 1, 16]
                score = torch.div(qkt, math.sqrt(self.hidden_dim))
            elif self.score_function == 'mlp':
                kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
                qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
                kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
                score = F.tanh(torch.matmul(kq, self.weight))
            elif self.score_function == 'bi_linear':
                qw = torch.matmul(qx, self.weight)
                kt = kx.permute(0, 2, 1)
                score = torch.bmm(qw, kt)
            else:
                raise RuntimeError('invalid score_function')
            score = F.softmax(score, dim=-1)
            out = torch.matmul(score, kx)  # [256, 1, 128]
            out_list.append(out)

        output = torch.cat(out_list, dim=-1)
        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        return output

class M_dimMultiHead_Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='scaled_dot_product', dropout=0):
        super(M_dimMultiHead_Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_kx = nn.Parameter(torch.FloatTensor(1, embed_dim, embed_dim))
        self.w_qx = nn.Parameter(torch.FloatTensor(1, embed_dim, embed_dim))
        self.Lin = nn.Linear(hidden_dim, 1, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim*2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters() # 对参数进行预处理
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.embed_dim)
        self.w_kx.data.uniform_(-stdv, stdv).renorm_(2, 1, 1e-5).mul_(1e5)
        self.w_qx.data.uniform_(-stdv, stdv).renorm_(2, 1, 1e-5).mul_(1e5)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, k, q):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
        # 类似于获取 特征和查询向量的隐藏表示
        kx = k.view(1, -1, self.embed_dim)
        kx = torch.tanh(torch.bmm(kx, self.w_kx).view(-1, k_len, self.embed_dim))
        qx = q.view(1, -1, self.embed_dim)  # 类似于全连接
        qx = torch.tanh(torch.bmm(qx, self.w_qx).view(-1, q_len, self.embed_dim))

        k_nhead = kx.view(mb_size, k_len, self.n_head, self.hidden_dim).permute(0, 2, 1, 3).contiguous()  # [256,2,21,128]
        kt = k_nhead.permute(0, 1, 3, 2).contiguous()  # [256,2,128,21]
        qx = qx.view(mb_size, q_len, self.n_head, self.hidden_dim).permute(0, 2, 1, 3).contiguous()  # [256,2,1,128]
        # 哈达玛积的计算
        memory = torch.mul(qx, k_nhead)
        score = torch.tanh(self.Lin(memory)) # [256, 2, 21, 1]
        score = score.permute(0, 1, 3, 2).contiguous()
        score = torch.div(score, math.sqrt(self.hidden_dim)) # [256, 2, 1, 21]
        score = F.softmax(score, dim=-1)
        output = torch.matmul(score, k_nhead).permute(0, 2, 1, 3).contiguous().view(mb_size, q_len, self.n_head * self.hidden_dim)
        output = self.proj(output)  # [256, 1, 256]
        output = self.dropout(output)
        return output

class Self_Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='scaled_dot_product', dropout=0):
        super(Self_Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_kx = nn.Parameter(torch.FloatTensor(n_head, embed_dim, hidden_dim))
        self.w_qx = nn.Parameter(torch.FloatTensor(n_head, embed_dim, hidden_dim))
        # self.Lin_k = nn.Linear(embed_dim, hidden_dim, bias=True)
        # self.Lin_q = nn.Linear(embed_dim, hidden_dim, bias=True)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim*2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters() # 对参数进行预处理

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        self.w_kx.data.uniform_(-stdv, stdv).renorm_(2, 1, 1e-5).mul_(1e5)
        self.w_qx.data.uniform_(-stdv, stdv).renorm_(2, 1, 1e-5).mul_(1e5)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, k, q, flag):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
        # 类似于获取 特征和查询向量的隐藏表示
        kx = k.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.embed_dim)
        kx = torch.tanh(torch.bmm(kx, self.w_kx).view(-1, k_len, self.hidden_dim))
        if flag == True:
            qx = q
        else:
            qx = q.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.embed_dim)  # 类似于全连接
            qx = torch.tanh(torch.bmm(qx, self.w_qx).view(-1, q_len, self.hidden_dim))
            # (n_head*?, q_len, hidden_dim)
        # kx = torch.tanh(self.Lin_k(k))
        # qx = torch.tanh(self.Lin_q(q))
        if self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1).contiguous()
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            score = F.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=-1)
        output = torch.bmm(score, kx)  # (n_head*?, q_len, hidden_dim)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (?, q_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        return output
class M_Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', dropout=0):
        super(M_Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_kx = nn.Parameter(torch.FloatTensor(n_head, embed_dim, hidden_dim))
        self.w_qx = nn.Parameter(torch.FloatTensor(n_head, embed_dim, hidden_dim))
        self.Lin_k = nn.Linear(embed_dim, hidden_dim, bias=True)
        self.Lin_q = nn.Linear(embed_dim, hidden_dim, bias=True)
        self.proj = nn.Linear(n_head * hidden_dim, 1, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim*2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters() # 对参数进行预处理

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        self.w_kx.data.uniform_(-stdv, stdv).renorm_(2, 1, 1e-5).mul_(1e5)
        self.w_qx.data.uniform_(-stdv, stdv).renorm_(2, 1, 1e-5).mul_(1e5)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, k, q):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
        # 类似于获取 特征和查询向量的隐藏表示
        kx_source = k
        kx = k.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.embed_dim)
        qx = q.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.embed_dim)  # (n_head, ?*q_len, embed_dim)
        kx = torch.sigmoid(torch.bmm(kx, self.w_kx).view(-1, k_len, self.hidden_dim))  # (n_head*?, k_len, hidden_dim) 类似于全连接
        qx = torch.sigmoid(torch.bmm(qx, self.w_qx).view(-1, q_len, self.hidden_dim))  # (n_head*?, q_len, hidden_dim)
        # kx = torch.tanh(self.Lin_k(k))
        # qx = torch.tanh(self.Lin_q(q))
        # 哈达玛积的计算
        memory = torch.mul(qx, kx)
        score = torch.sigmoid(self.proj(memory))
        score = score.permute(0, 2, 1).contiguous()
        score = torch.div(score, math.sqrt(self.hidden_dim))
        # if self.score_function == 'scaled_dot_product':
        #     kt = kx.permute(0, 2, 1).contiguous()
        #     t = kt.is_contiguous()
        #     qkt = torch.bmm(qx, kt)
        #     score = torch.div(qkt, math.sqrt(self.hidden_dim))
        # elif self.score_function == 'mlp':
        #     kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
        #     qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
        #     kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
        #     score = F.tanh(torch.matmul(kq, self.weight))
        # elif self.score_function == 'bi_linear':
        #     qw = torch.matmul(qx, self.weight)
        #     kt = kx.permute(0, 2, 1)
        #     score = torch.bmm(qw, kt)
        # else:
        #     raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=-1)
        output = torch.bmm(score, kx_source)  # (n_head*?, q_len, hidden_dim)
        # output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (?, q_len, n_head*hidden_dim)
        # output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        return output


class CosLayer(nn.Module):
    def __init__(self, in_size, out_size, s=4.6):
        super(CosLayer, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.W = nn.Parameter(torch.randn(out_size, in_size))
        self.W.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.s = nn.Parameter(torch.randn(1,)) if s is None else s

    def forward(self, input):
        cosine = F.linear(F.normalize(input), F.normalize(self.W))
        output = cosine * self.s
        return output

    def __repr__(self):
        return self.__class__.__name__ +  '(in_size={}, out_size={}, s={})'.format(
                    self.in_size, self.out_size,
                    'learn' if isinstance(self.s, nn.Parameter) else self.s)
class Interaction(nn.Module):
    def __init__(self, hidden_dim, l, batch_size, n_classes):
        super(Interaction, self).__init__()
        self.l = l
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.b_size = batch_size
        stdv = 1. / math.sqrt(self.hidden_dim)
        # 在处理 Xj 时出现问题，Xj是二维向量，使用softmax不能将其转化为一个数，所以有两个考虑：
        #1，全程是q     2，使用线性层将其转化为一个数（目前采用）
        self.qd = nn.Parameter(torch.randn(1, self.hidden_dim))
        self.qd.data.uniform_(-stdv, stdv).renorm_(2, 0, 1e-5).mul_(1e5)
        self.qp_1 = nn.Parameter(torch.randn(1, self.hidden_dim))
        self.qp_1.data.uniform_(-stdv, stdv).renorm_(2, 0, 1e-5).mul_(1e5)
        self.qp_2 = nn.Parameter(torch.randn(1, self.hidden_dim))
        self.qp_2.data.uniform_(-stdv, stdv).renorm_(2, 0, 1e-5).mul_(1e5)
        self.qr = nn.Parameter(torch.randn(1, self.hidden_dim))
        self.qr.data.uniform_(-stdv, stdv).renorm_(2, 0, 1e-5).mul_(1e5)

        # 交互学习模型
        self.D_Self_A = Self_dimMultiHead_Attention1(self.hidden_dim, None, 256, 2, score_function='scaled_dot_product', dropout=0)
        self.P_Self_A_1 = Self_dimMultiHead_Attention1(self.hidden_dim, None, 256, 2, score_function='scaled_dot_product', dropout=0)
        self.P_Self_A_2 = Self_dimMultiHead_Attention1(self.hidden_dim, None, 256, 2, score_function='scaled_dot_product', dropout=0)
        self.R_Self_A = Self_dimMultiHead_Attention1(self.hidden_dim, None, 256, 2, score_function='scaled_dot_product', dropout=0)
        # self.P_Self_A = Self_dimMultiHead_Attention(self.hidden_dim, self.hidden_dim // 2, 256, 2, score_function='scaled_dot_product', dropout=0)

        self.PgD_M_A = M_Attention(self.hidden_dim, score_function='dot_product', dropout=0)
        self.DgP_M_A = M_Attention(self.hidden_dim, score_function='dot_product', dropout=0)
        self.RgP_M_A = M_Attention(self.hidden_dim, score_function='dot_product', dropout=0)
        self.PgR_M_A = M_Attention(self.hidden_dim, score_function='dot_product', dropout=0)
        # self.RgP_M_A = M_Attention(self.hidden_dim, score_function='dot_product', dropout=0)

        self.gru_cell_D = nn.GRUCell(self.hidden_dim, self.hidden_dim, bias=True)
        self.gru_cell_P_1 = nn.GRUCell(self.hidden_dim, self.hidden_dim, bias=True)
        self.gru_cell_P_2 = nn.GRUCell(self.hidden_dim, self.hidden_dim, bias=True)
        self.gru_cell_R = nn.GRUCell(self.hidden_dim, self.hidden_dim, bias=True)
        # 线性层（4层）
        self.Linear1 = nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2)
        self.Linear2 = CosLayer(self.hidden_dim * 2, 21)
        self.Linear3 = nn.Linear(21, 21)
        self.Linear4 = nn.Linear(21, 21)
        self.Linear5 = nn.Linear(42, 21)
        # 残差连接
        self.res_con1 = nn.Sequential(nn.Conv1d(self.hidden_dim * 2, self.hidden_dim,kernel_size=1),
                                      nn.BatchNorm1d(256),
                                      nn.ReLU())
        self.res_con2 = nn.Sequential(nn.Conv1d(self.hidden_dim * 2, self.hidden_dim,kernel_size=1),
                                      nn.BatchNorm1d(256),
                                      nn.ReLU())
        self.res_con3 = nn.Sequential(nn.Conv1d(self.hidden_dim * 2, self.hidden_dim,kernel_size=1),
                                      nn.BatchNorm1d(256),
                                      nn.ReLU())
        # 卷积池化操作
        self.conv = nn.Conv1d(3, 3, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.norm_bn = nn.BatchNorm1d(self.hidden_dim)

        self.norm_bn4 = nn.BatchNorm1d(3)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))
        self.mean_pool = nn.AvgPool2d(kernel_size=(3, 1), stride=(3, 1))


    def forward(self, D_g, P_g, R_g):
        # 将批次图数据转为一个个的图数据
        D_singel = dgl.unbatch(D_g)
        P_singel = dgl.unbatch(P_g)
        R_singel = dgl.unbatch(R_g)
        mb_size = len(D_singel)
        # 变成大小为 [256, 21, 256]的张量
        D_node_feature = torch.stack([D_singel[i].ndata['x'] for i in range(0, mb_size)])
        P_node_feature = torch.stack([P_singel[k].ndata['x'] for k in range(0, mb_size)])
        R_node_feature = torch.stack([R_singel[j].ndata['x'] for j in range(0, mb_size)])
        # #残差连接
        D_max_pool = dgl.max_nodes(D_g, 'x')
        D_mean_pool = dgl.mean_nodes(D_g, 'x')
        D_pool = self.res_con1(torch.cat((D_max_pool,D_mean_pool), dim=1).unsqueeze(2)).squeeze(2)

        P_max_pool = dgl.max_nodes(P_g, 'x')
        P_mean_pool = dgl.mean_nodes(P_g, 'x')
        P_pool = self.res_con3(torch.cat((P_max_pool, P_mean_pool), dim=1).unsqueeze(2)).squeeze(2)

        R_max_pool = dgl.max_nodes(R_g, 'x')
        R_mean_pool = dgl.mean_nodes(R_g, 'x')
        R_pool = self.res_con2(torch.cat((R_max_pool,R_mean_pool), dim=1).unsqueeze(2)).squeeze(2)

        # 一般注意
        qd = self.qd.repeat(mb_size, 1)
        qp_1 = self.qp_1.repeat(mb_size, 1)
        qp_2 = self.qp_2.repeat(mb_size, 1)
        qr = self.qr.repeat(mb_size, 1)

        flag_onelayer = False
        VD = self.D_Self_A(D_node_feature, qd, flag_onelayer).squeeze(dim=1)
        VP_1 =self.P_Self_A_1(P_node_feature, qp_1, flag_onelayer).squeeze(dim=1)
        VP_2 = self.P_Self_A_2(P_node_feature, qp_2, flag_onelayer).squeeze(dim=1)
        VR = self.R_Self_A(R_node_feature, qr, flag_onelayer).squeeze(dim=1)

        # 交互学习
        layers = self.l - 1
        for _ in range(0, layers):
            Atten_P2D = self.PgD_M_A(D_node_feature, VP_1).squeeze(dim=1)  # 跨模态注意
            Atten_D2D = self.D_Self_A(D_node_feature, VD, flag_onelayer).squeeze(dim=1)  # 自注意

            Atten_D2P = self.DgP_M_A(P_node_feature, VD).squeeze(dim=1)
            Atten_P2P_1 = self.P_Self_A_1(P_node_feature, VP_1, flag_onelayer).squeeze(dim=1)

            Atten_R2P = self.RgP_M_A(P_node_feature, VR).squeeze(dim=1)
            Atten_P2P_2 = self.P_Self_A_2(P_node_feature, VP_2, flag_onelayer).squeeze(dim=1)

            Atten_P2R = self.PgR_M_A(R_node_feature, VP_2).squeeze(dim=1)
            Atten_R2R = self.R_Self_A(R_node_feature, VR, flag_onelayer).squeeze(dim=1)

            Atten_D = self.gru_cell_D(Atten_P2D, Atten_D2D)
            Atten_P_1 = self.gru_cell_P_1(Atten_D2P, Atten_P2P_1)
            Atten_P_2 = self.gru_cell_P_2(Atten_R2P, Atten_P2P_2)
            Atten_R = self.gru_cell_R(Atten_P2R, Atten_R2R)

            VD = self.relu(self.norm_bn(self.gru_cell_D(Atten_D, VD)))
            VP_1 = self.relu(self.norm_bn(self.gru_cell_P_1(Atten_P_1, VP_1)))
            VP_2 = self.relu(self.norm_bn(self.gru_cell_P_2(Atten_P_2, VP_2)))
            VR = self.relu(self.norm_bn(self.gru_cell_R(Atten_R, VR)))

        # 残差连接
        VD = VD + D_pool
        VP_1 = VP_1 + P_pool
        VP_2 = VP_2 + P_pool
        VR = VR + R_pool
        # 拼接操作
        Km = torch.cat((VD, VP_1, VP_2, VR), dim=1)
        # 线性层(没加激活函数)
        FC1 = self.Linear1(Km)
        FC2 = self.Linear2(FC1)
        FC3 = self.Linear3(FC2)
        FC4 = self.Linear4(FC3)
        # output = FC2

        # 将经过线性层的张亮改为三通道
        Fm = torch.cat((FC2.unsqueeze(1), FC3.unsqueeze(1), FC4.unsqueeze(1)), dim=1)
        r_conv = F.relu(self.norm_bn4(self.conv(Fm)))
        # 最大池化
        pool_max = self.max_pool(r_conv.unsqueeze(1)).squeeze(1).squeeze(1)
        pool_mean = self.mean_pool(r_conv.unsqueeze(1)).squeeze(1).squeeze(1)
        output = torch.cat((pool_mean, pool_max), dim=1)
        output = self.Linear5(output)
        return output

class EIMANN(nn.Module):
    def __init__(self,Extract_dim, Interaction_dim, l, Batch_size, n_classes):
        super(EIMANN, self).__init__()
        self.Extract_dim = Extract_dim
        self.Interaction_dim = Interaction_dim
        self.b_size = Batch_size
        self.enn = ExtractNetwork(self.Extract_dim)
        self.interaction = Interaction(Interaction_dim, l, self.b_size, n_classes)

    def forward(self,D_x, R_x, P_x):
        D_g = self.enn(D_x)
        P_g = self.enn(P_x)
        R_g = self.enn(R_x)
        Output = self.interaction(D_g, P_g, R_g)
        return Output