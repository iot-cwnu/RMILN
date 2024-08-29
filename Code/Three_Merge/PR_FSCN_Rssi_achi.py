import os
import math
import dgl
from torch.utils.data import Dataset
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from PR_FSCN_Rssi import RFDataset
from Interaction_Rssi import EIMANN
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score
import time
device = torch.device('cuda')
filenames = os.listdir(r'../15_16node')
alldata = []
labeldata = []
fenge = []
ztfenge = []
depth = 1
BATCH_SIZE = 64
label_tem = 0

# 21个节点
# mindf: -1106.6875
# maxdf: 963.5
# minRssi: -81.5
# maxRssi: 0
# minPhase: 0
# maxPhase: 6.280117345603815

# 16个节点
# maxdf: 1162.875
# mindf: -1449.3125
# maxRssi: 0
# minRssi: -81.5
# maxPhase: 6.280117345603815
# minPhase: 0

def getdata(path):
    filenames = os.listdir(path)
    alllabel = []
    for i in range(len(filenames)):
        zpath = '../15_16node/' + filenames[i]
        with open(zpath, "r") as f:
            lines=f.readlines()
            newdata = []
            for id in range(0, 16):
                newdata.append([])
            for i in range(len(lines)):
                line = lines[i] .strip('\n')
                x = line.split()
                if len(x)<7:
                    alllabel.append(label_tem)
                    alldata.append(newdata)
                    newdata = []
                    for id in range(0, 16):
                        newdata.append([])
                if len(x) == 7:
                    df = float(x[3])
                    rssi = float(x[4])
                    phase = float(x[5])
                    if df > 0:
                        df = df / 963.5
                    elif df < 0:
                        df = df / 1106.6875
                    else:
                        df = df
                    id = int(x[1])
                    newdata[id].append(df)  # DF
                    newdata[id].append(rssi / (-81.5))  # RSSI
                    newdata[id].append(phase / 6.280117345603815)  # Phase
                    label_tem = int(x[6])
        # break
    print(len(alllabel), len(alldata))

    allalldata = []
    for j in alldata:
        for q in range(len(j)):
            if len(j[q]) < (180 // depth):
                x = 180 // depth - len(j[q])
                buchong = [0] * x
                j[q] = j[q] + buchong
            if len(j[q]) > 180 // depth:
                j[q] = j[q][:180 // depth]
        allalldata.append(j)
    return allalldata, alllabel

data, label = getdata('../15_16node')
print(len(data))
print(len(data[0]))
print(len(data[0][0]))

import random
def split_train_test(data, label, test_ratio):
    random.seed(38)
    random.shuffle(data)
    random.seed(38)
    random.shuffle(label)
    test_set_size = int(len(data) * test_ratio)
    test_data = data[:test_set_size]
    test_label = label[:test_set_size]
    train_data = data[test_set_size:]
    train_label = label[test_set_size:]
    return train_data, train_label, test_data, test_label

traindata, trainlabel, testdata, testlabel = split_train_test(data, label, 0.2)


EPOCH = 1
TIME_STEP = 28
INPUT_SIZE = 28
# LR = 0.0015  LR = 0.0020
LR = 0.002

print(len(testlabel))

#创建时间空间特征提取网络，改下优化器

eimAMN = EIMANN(256, 256, 3, BATCH_SIZE, 21).to(device)
optimizer = torch.optim.Adam(eimAMN.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1)
# scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.1, total_iters=45)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.001)
loss_func = nn.CrossEntropyLoss()

#获得三种骨架活动图数据集
def get_Set(X_data,X_label):
    D_X_data = []
    R_X_data = []
    P_X_data = []
    for data in X_data:
        D_data = []
        R_data = []
        P_data = []
        for id in range(0, 16):
            D_data.append([])
            R_data.append([])
            P_data.append([])
        for i in range(0, len(data)):
            for j in range(0, len(data[i]), 3):
                D_data[i].append(data[i][j])
                R_data[i].append(data[i][j+1])
                P_data[i].append(data[i][j+2])
        D_X_data.append(D_data)
        R_X_data.append(R_data)
        P_X_data.append(P_data)
    return D_X_data, R_X_data, P_X_data, X_label

D_traindata, R_traindata, P_traindata, train_label = get_Set(traindata, trainlabel)
D_testdata, R_testdata, P_testdata, test_label = get_Set(testdata, testlabel)
# print(len(D_traindata),len(train_label))
# print(len(D_testdata),len(test_label))

# 获得训练数据集和测试数据集
D_trainset = RFDataset(D_traindata,train_label)# 返回图和标签的列表
R_trainset = RFDataset(R_traindata,train_label)
P_trainset = RFDataset(P_traindata,train_label)

D_testset = RFDataset(D_testdata,test_label)
R_testset = RFDataset(R_testdata,test_label)
P_testset = RFDataset(P_testdata,test_label)
print(D_trainset)


def get_confusion_matrix(preds, labels, num_classes, normalize="true"):
    if isinstance(preds, list):
        preds = torch.cat(preds, dim=0)
    if isinstance(labels, list):
        labels = torch.cat(labels, dim=0)
    if labels.ndim == preds.ndim:
        labels = torch.argmax(labels, dim=-1)
    preds = torch.flatten(torch.argmax(preds, dim=-1))
    labels = torch.flatten(labels)
    # 混淆矩阵的标签 labels 应该是一个类别标签。
    cmtx = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    print(cmtx)

    # 准确率 normalize=False则返回做对的个数
    acc = accuracy_score(labels, preds)
    acc_nums = accuracy_score(labels, preds, normalize=False)
    print('\nhx_acc:{}, preds:{}'.format(acc, acc_nums))

    # labels为指定标签类别，average为指定计算模式

    # micro-precision
    micro_p = precision_score(labels, preds, labels=list(range(num_classes)), average='micro')
    # micro-recall
    micro_r = recall_score(labels, preds, labels=list(range(num_classes)), average='micro')
    # micro f1-score
    micro_f1 = f1_score(labels, preds, labels=list(range(num_classes)), average='micro')

    print('micro_p:{}  micro_r:{}  micro_f1:{}'.format(micro_p, micro_r, micro_f1))

    # macro-precision
    macro_p = precision_score(labels, preds, labels=list(range(num_classes)), average='macro')
    # macro-recall
    macro_r = recall_score(labels, preds, labels=list(range(num_classes)), average='macro')
    # macro f1-score
    macro_f1 = f1_score(labels, preds, labels=list(range(num_classes)), average='macro')

    print('macro_p:{}  macro_r:{}  macro_f1:{}\n'.format(macro_p, macro_r, macro_f1))
    return cmtx


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    return dgl.batch(graphs), torch.tensor(labels, dtype=torch.long)

D_train_loader = torch.utils.data.DataLoader(dataset=D_trainset, batch_size=BATCH_SIZE, collate_fn=collate)
R_train_loader = torch.utils.data.DataLoader(dataset=R_trainset, batch_size=BATCH_SIZE, collate_fn=collate)
P_train_loader = torch.utils.data.DataLoader(dataset=P_trainset, batch_size=BATCH_SIZE, collate_fn=collate)

writer = SummaryWriter('../Three_Merge_result/Batch_size/logs_Rssi64')

def train(ep):
    train_loss = 0.0
    for step, (D, R, P) in enumerate(zip(D_train_loader, R_train_loader, P_train_loader)):
        D_x, D_label = D
        R_x, R_label = R
        P_x, P_label = P
        D_x, R_x, P_x, label = D_x.to(device), R_x.to(device), P_x.to(device), D_label.to(device)
        # print(D_x)
        # print("--------------------------------")
        optimizer.zero_grad()
        Output = eimAMN(D_x, R_x, P_x)
        loss = loss_func(Output, label)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    loss = train_loss / len(D_train_loader.dataset)
    writer.add_scalar('Loss/train', loss, global_step=ep)
    for name, param in eimAMN.named_parameters():
        if param.requires_grad and param.grad is not None:
            # 使用add_histogram来记录梯度
            writer.add_histogram(f'gradients/{name}', param.grad.data.cpu().numpy(), global_step=ep)
            # 记录权值的直方图
            writer.add_histogram(f'weights/{name}', param.data.cpu().numpy(), global_step=ep)
    print('ok')


acclist=[]
cmtxlist=[]

best_acc_list = []
def test(ep):
    test_loss = 0
    correct = 0
    preds = []
    labels = []
    D_test_loader = torch.utils.data.DataLoader(dataset=D_testset, batch_size=BATCH_SIZE, collate_fn=collate)
    R_test_loader = torch.utils.data.DataLoader(dataset=R_testset, batch_size=BATCH_SIZE, collate_fn=collate)
    P_test_loader = torch.utils.data.DataLoader(dataset=P_testset, batch_size=BATCH_SIZE, collate_fn=collate)
    with torch.no_grad():
        for step, (D, R, P) in enumerate(zip(D_test_loader, R_test_loader, P_test_loader)):
            D_x, D_label = D
            R_x, R_label = R
            P_x, P_label = P
            D_label = torch.as_tensor(D_label, dtype=torch.int64)
            D_x, R_x, P_x, label = D_x.to(device), R_x.to(device), P_x.to(device), D_label.to(device)
            output = eimAMN(D_x, R_x, P_x)
            preds.append(output.cpu())
            labels.append(label.cpu())
            test_loss += loss_func(output, label).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        test_loss /= len(D_test_loader.dataset)
        writer.add_scalar('Loss/test', test_loss, global_step=ep)
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        cmtx = get_confusion_matrix(preds, labels, 21)
        # print(cmtx)
        for i in range(len(cmtx)):
            print('the acc of activity {}: {}'.format(i,  cmtx[i][i]/sum(cmtx[i])))
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
            test_loss, correct, len(D_test_loader.dataset),
            100. * correct / len(D_test_loader.dataset)))
        acc = correct / len(D_test_loader.dataset)
        writer.add_scalar("test accuary", acc, global_step=ep)
        acclist.append(float(acc))
        cmtxlist.append(cmtx.tolist())
        # 保存最好的模型参数
        if ep == 0:
            best_acc_list.append(acc)
            torch.save(eimAMN.state_dict(), '../Three_Merge_result/Batch_size/Rssi64.mdl')
        else:
            if acc > max(best_acc_list):
                torch.save(eimAMN.state_dict(), '../Three_Merge_result/Batch_size/Rssi64.mdl')
                best_acc_list.append(acc)

        if ep == 55:
            with open("../Three_Merge_result/Batch_size/Rssi64.txt", "a+") as f:
                f.write('\n')
                # f.write('type={}:'.format(type_X))
                f.write(str(acclist))
                f.write('\n')
                f.write(str(max(acclist)))
                f.write(str(cmtxlist[acclist.index(max(acclist))]))
                f.write('\n')
        print('acc:',acclist)
        print('the max of the acc:',max(acclist))
        return test_loss

if __name__ == '__main__':
    for epoch in range(0, 56):
        print('Epoch:',epoch)
        train(epoch)
        if epoch == 1:
            model_dict = torch.load('../Three_Merge_result/Batch_size/Rssi64.mdl')
            dict_name = list(model_dict)
            for i, p in enumerate(dict_name):
                print(i, p)
        Stime = time.time()
        test(epoch)
        Etime = time.time()
        print("rtime=", (Etime-Stime)/4)
        if epoch % 20 == 0:
            LR /= 10
        # scheduler.step()
        # break
    print('batch_size:', BATCH_SIZE)
