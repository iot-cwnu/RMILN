import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
depth=1
adj_filename="../data/adj16.txt"
class RFDataset(Dataset):
    def __init__(self, data,label,repeat=1, num_classes=21):
        self.graph_label_list = self.read_file(data,label)
        self.len = len(self.graph_label_list)
        self.repeat = repeat
        self.num_classes = num_classes

    def __getitem__(self, i):
        graph, label = self.graph_label_list[i]
        return graph, label
    def __len__(self):
        data_len = len(self.graph_label_list) * self.repeat
        return data_len
    def create_graph(self, feature):
        i = self.load_adj()
        g = dgl.graph(i)
        g.ndata['x'] = feature
        g.edata['x'] = torch.ones(g.num_edges(), dtype=torch.int)
        return g

    def load_adj(self):
        start_node=list()
        end_node=list()
        with open(adj_filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                content = line.rstrip().split(',')
                start_node.append(int(content[0]))
                end_node.append(int(content[1]))
        return torch.tensor(start_node),torch.tensor(end_node)

    def read_file(self,data,label):
        graph_label_list = []
        for i in range(0,len(data)):
            feature=[]
            for j in range(len(data[i])):
                feature.append(data[i][j])
            features = torch.tensor(feature, dtype=torch.float32)
            g = self.create_graph(features)
            if i==len(data)-1:
                print(features.shape)
            l=label[i]
            graph_label_list.append((g, l))
        return graph_label_list

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

class ExtractNetwork(nn.Module):
    def __init__(self, hidden_dim):
        super(ExtractNetwork, self).__init__()
        self.fusion1 = nn.Sequential(nn.Conv1d(1, 32, kernel_size=5, stride=5),
                                     nn.BatchNorm1d(32),
                                     nn.ReLU(inplace=True))
        self.fusion2 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=3, stride=3),
                                     nn.BatchNorm1d(64),
                                     nn.ReLU(inplace=True))

        self.conv1 = dgl.nn.pytorch.GATConv(hidden_dim, hidden_dim // 2, 2)
        self.conv2 = dgl.nn.pytorch.GATConv(hidden_dim, hidden_dim // 2, 2)

        self.data_bn = nn.BatchNorm1d(hidden_dim * 3)
        self.cos = CosLayer(hidden_dim * 3, hidden_dim)


    def forward(self, g):
        # 进行时间卷积
        h = g.ndata['x']
        h = torch.unsqueeze(h, dim=1)
        # h = self.fusion(h).contiguous()
        h = self.fusion1(h)
        h = self.fusion2(h).contiguous()

        # RC模块
        N, C, T= h.size()
        h = h.view(N, C* T,1)
        h=torch.squeeze(h,dim=2)
        g.ndata['x'] = h

        # 第一层图卷积
        h11 = F.relu(self.conv1(g, h))
        h12 = torch.tanh(self.conv1(g, h))
        h13 = F.logsigmoid(self.conv1(g, h))

        # ASFM模块
        h1 = h11+ h12+h13
        # g.ndata['x'] = h11
        # h1 = g.ndata['x']
        N, C, T = h1.size()
        h1 = h1.view(N, C*T)
        g.ndata['x'] = h1

        # 第二三层图卷积
        h2 = torch.tanh(self.conv2(g, h1))
        N, C, T = h2.size()
        h2 = h2.view(N, C * T)
        g.ndata['x'] = h2

        h3 = F.relu(self.conv2(g, h2))
        N, C, T = h3.size()
        h3 = h3.view(N, C * T)

        # h1,h2,h3: [256*21,512]
        # SOC模块
        h_c = torch.cat((h1, h2, h3), dim=1)
        h_c = self.cos(self.data_bn(h_c))
        g.ndata['x'] = h_c
        return g
