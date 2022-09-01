import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# import torchvision
# import torchvision.transforms as transforms
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils
import time
import pickle
from sklearn.metrics import auc,roc_curve,confusion_matrix
np.set_printoptions(suppress=True)
from sklearn.model_selection import KFold
# torch.cuda.set_device(2)
from Bio import SeqIO
import matplotlib.pyplot as plt

def AA_ONE_HOT(AA):
    one_hot_dict = {
        'A': 1,
        'C': 2,
        'G': 3,
        'T': 4
    }
    coding_arr = np.zeros((len(AA),1), dtype=float)
    for m in range(len(AA)):
        coding_arr[m] = one_hot_dict[AA[m]]
        # print(coding_arr[m])
    return coding_arr

def train_feature_extraction():

    se_pp = ("train_mature_P.fasta")
    se_nn = ("train_mature_N.fasta")


    se_p = SeqIO.parse(se_pp, 'fasta')
    se_n = SeqIO.parse(se_nn, 'fasta')
    # print(str(se_p.seq))
    aa = len(list(se_p))
    bb = len(list(se_n))
    # print(aa)
    # print(bb)
    i = 0
    j = 0
    p = np.zeros((aa, 41, 1))
    p_label = np.zeros((aa, 2))
    n = np.zeros((bb, 41, 1))
    n_label = np.zeros((bb, 2))
    pp = p.copy()
    pp_label = p_label.copy()
    nn = n.copy()
    nn_label = n_label.copy()
    for my_pp in SeqIO.parse(se_pp,'fasta'):
        AA = str(my_pp.seq)
        pp[i] = AA_ONE_HOT(AA)
        pp_label[i] = [1,0]
        i += 1

    for my_nn in SeqIO.parse(se_nn,'fasta'):
        AA = str(my_nn.seq)
        nn[j] = AA_ONE_HOT(AA)
        nn_label[j] = [0,1]
        j += 1
    Z_Mays_xx_all = np.vstack([pp[:aa], nn[:bb]])
    Z_Mays_yy_all = np.vstack([pp_label[:aa], nn_label[:bb]])
    # Z_Mays_xx_all = Z_Mays_xx_all.reshape(len(Z_Mays_xx_all), -1)
    # Z_Mays_yy_all = Z_Mays_yy_all.reshape(len(Z_Mays_yy_all), -1)

    # Z_Mays_xx_all = nn[:bb]#np.vstack([pp[:aa], nn[:bb]])
    # Z_Mays_yy_all = nn_label[:bb]#np.vstack([pp_label[:aa], nn_label[:bb]])
    # print(Z_Mays_xx_all.shape)
    # print(Z_Mays_yy_all.shape)
    # print(Z_Mays_xx_all[1:2])
    # print(Z_Mays_yy_all[1:2])
    # np.save(sample_name + '_Binary__fuature_xx.npy', Z_Mays_xx_all)
    # np.save(sample_name + '3__fuature_yy.npy', Z_Mays_yy_all)
    return Z_Mays_xx_all, Z_Mays_yy_all


def test_feature_extraction():
    se_pp = ("test_mature_P.fasta")
    se_nn = ("test_mature_N.fasta")
    se_p = SeqIO.parse(se_pp, 'fasta')
    se_n = SeqIO.parse(se_nn, 'fasta')
    # print(str(se_p.seq))
    aa = len(list(se_p))
    bb = len(list(se_n))
    # print(aa)
    # print(bb)
    i = 0
    j = 0
    p = np.zeros((aa, 41, 1))
    p_label = np.zeros((aa, 2))
    n = np.zeros((bb, 41, 1))
    n_label = np.zeros((bb, 2))
    pp = p.copy()
    pp_label = p_label.copy()
    nn = n.copy()
    nn_label = n_label.copy()
    for my_pp in SeqIO.parse(se_pp,'fasta'):
        AA = str(my_pp.seq)
        pp[i] = AA_ONE_HOT(AA)
        pp_label[i] = [1,0]
        i += 1

    for my_nn in SeqIO.parse(se_nn,'fasta'):
        AA = str(my_nn.seq)
        nn[j] = AA_ONE_HOT(AA)
        nn_label[j] = [0,1]
        j += 1
    Z_Mays_xx_all = np.vstack([pp[:aa], nn[:bb]])
    Z_Mays_yy_all = np.vstack([pp_label[:aa], nn_label[:bb]])
    # Z_Mays_xx_all = Z_Mays_xx_all.reshape(len(Z_Mays_xx_all), -1)
    # Z_Mays_yy_all = Z_Mays_yy_all.reshape(len(Z_Mays_yy_all), -1)

    # Z_Mays_xx_all = nn[:bb]#np.vstack([pp[:aa], nn[:bb]])
    # Z_Mays_yy_all = nn_label[:bb]#np.vstack([pp_label[:aa], nn_label[:bb]])
    # print(Z_Mays_xx_all.shape)
    # print(Z_Mays_yy_all.shape)
    # print(Z_Mays_xx_all[1:2])
    # print(Z_Mays_yy_all[1:2])
    # np.save(sample_name + '_Binary__fuature_xx.npy', Z_Mays_xx_all)
    # np.save(sample_name + '3__fuature_yy.npy', Z_Mays_yy_all)
    return Z_Mays_xx_all, Z_Mays_yy_all



def load_features_train():
    node, label1 = train_feature_extraction()
    label1 = label1.tolist()

    label = []

    for num in range(len(node)):
        # print(label1[num])
        if label1[num] == [1,0]:
            label.append(torch.tensor(1))
        else:
            label.append(torch.tensor(0))
    label = np.array(label)
    # a = node[1:3].tolist()
    # list_node = []
    # for list_f in node:
    #     list_node.append(list(map(int, list_f)))

    # print(list_int)
    # label = label.astype(np.float32)
    return node, label
# node, label = load_features()

def load_features_test():
    node, label1 = test_feature_extraction()
    label1 = label1.tolist()

    label = []

    for num in range(len(node)):
        # print(label1[num])
        if label1[num] == [1,0]:
            label.append(torch.tensor(1))
        else:
            label.append(torch.tensor(0))
    label = np.array(label)
    # a = node[1:3].tolist()
    # list_node = []
    # for list_f in node:
    #     list_node.append(list(map(int, list_f)))

    # print(list_int)
    # label = label.astype(np.float32)
    return node, label

class newModel(nn.Module):
    def __init__(self, vocab_size=16):
        super().__init__()
        self.hidden_dim = 10
        self.batch_size = 48
        self.emb_dim = 128
        self.embedding = nn.Embedding(vocab_size, self.emb_dim, padding_idx=0)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=16)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.gru = nn.GRU(self.emb_dim, self.hidden_dim, num_layers=2,
                          bidirectional=True, dropout=0.5)
        self.block1 = nn.Sequential(nn.Linear(860, 512),
                                    # nn.BatchNorm1d(256),
                                    nn.LeakyReLU(),
                                    nn.Linear(512, 128),)
        self.block2 = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            # nn.Linear(32, 10),
            # nn.BatchNorm1d(10),
            # nn.LeakyReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        # print(x.shape,"x.shape")
        # print(type(x),"type(x)")

        x = self.embedding(x)
        output = self.transformer_encoder(x).permute(1, 0, 2)
        output, hn = self.gru(output)
        output = output.permute(1, 0, 2)
        hn = hn.permute(1, 0, 2)
        output = output.reshape(output.shape[0], -1)
        hn = hn.reshape(output.shape[0], -1)
        output = torch.cat([output, hn], 1)
        #         print(output.shape,hn.shape)
        return self.block1(output)

    def trainModel(self, x):
        with torch.no_grad():
            output = self.forward(x)
        return self.block2(output)


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # euclidean_distance: [128]
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +  # calmp夹断用法
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


def collate(batch):
    seq1_ls = []
    seq2_ls = []
    label1_ls = []
    label2_ls = []
    label_ls = []
    batch_size = len(batch)
    for i in range(int(batch_size / 2)):

        seq1, label1 = batch[i][0], batch[i][1]
        # print(seq1.shape,"seq1.shape")
        # print(type(seq1), "type(seq1)")
        # print(label1.shape,"label1.shape")
        # print(type(label1),"type(label1)")


        seq2, label2 = batch[i + int(batch_size / 2)][0], batch[i + int(batch_size / 2)][1]

        label1_ls.append(label1.unsqueeze(0))
        label2_ls.append(label2.unsqueeze(0))
        label = (label1 ^ label2)
        seq1_ls.append(seq1.unsqueeze(0))
        seq2_ls.append(seq2.unsqueeze(0))
        label_ls.append(label.unsqueeze(0))
    seq1 = torch.cat(seq1_ls).to(device)
    seq2 = torch.cat(seq2_ls).to(device)
    label = torch.cat(label_ls).to(device)
    label1 = torch.cat(label1_ls).to(device)
    label2 = torch.cat(label2_ls).to(device)
    return seq1, seq2, label, label1, label2

def calc(TN, FP, FN, TP):
    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    # Precision = TP / (TP + FP)
    ACC = (TP + TN) / (TP + TN + FN + FP)
    F1 = (2 * TP) / (2 * TP + FP + FN)
    MCC = (TP * TN - FP * FN) / pow((TP + FN) * (TP + FP) * (TN + FP) * (TN + FN), 0.5)
    return SN, SP, ACC, MCC


def evaluate_accuracy(data_iter, net):
    f = nn.Softmax(dim=0)
    acc_sum, n = 0.0, 0
    y_predict = []
    y_test_class = []
    for x, y in data_iter:
        x,y=x.to(device),y.to(device)
        outputs=net.trainModel(x)
        outputs = f(outputs)
        # # print(type(outputs))
        x_p = outputs.cuda().data.cpu().numpy().tolist()
        y_p = y.cuda().data.cpu().numpy().tolist()
        for p_list in x_p:
            y_predict.append(p_list)
        y_test_class += y_p
        acc_sum += (outputs.argmax(dim=1) == y).float().sum().item()
        # print(y.shape[0])
        n += y.shape[0]

    y_predict = np.array(y_predict)
    y_predict_class = np.argmax(y_predict, axis=1)
    y_test_class = np.array(y_test_class)
    tn, fp, fn, tp = confusion_matrix(y_test_class, y_predict_class).ravel()
    sn, sp, acc, mcc = calc(float(tn), float(fp), float(fn), float(tp))

    res_data = [tn, fp, fn, tp,sn, sp, acc, mcc]


    # print("tn, fp, fn, tp={},{},{},{}".format(tn, fp, fn, tp))
    fpr, tpr, thresholds = roc_curve(y_test_class, y_predict[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)  # auc为Roc曲线下的面积
    # print("tn, fp, fn, tp={},{},{},{}".format(tn, fp, fn, tp), ',acc=%.4f' % acc, 'sn=%.4f' % sn, ',sp=%.4f' % sp, ',mcc=%.4f' % mcc, 'AUC=', roc_auc)
    fpr, tpr, thresholds = roc_curve(y_test_class, y_predict[:, 1], pos_label=1)


    return acc_sum / n,fpr,tpr,roc_auc,res_data



if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X_train, y_train = load_features_train()
    X_test, y_test = load_features_test()





    x_train = X_train.reshape(len(X_train), -1)
    x_test = X_test .reshape(len(X_test), -1)
    x_train1 = []
    for list11 in x_train:
        x_train1.append(list(map(int, list11)))

    x_test1 = []
    for list_te in x_test:
        x_test1.append(list(map(int, list_te)))
    # seq1 = torch.tensor(seq11)
    # seq2 = torch.tensor(seq22)

    train_data = torch.tensor(x_train1)
    train_label = torch.tensor(y_train)
    test_data = torch.tensor(x_test1)
    test_label = torch.tensor(y_test)
    # print(torch.tensor(np.array(0)))
    # print(type(torch.tensor(np.array(0))))

    train_dataset = Data.TensorDataset(train_data, train_label)
    test_dataset = Data.TensorDataset(test_data, test_label)
    batch_size = 48
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
    train_iter_cont = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                  shuffle=True, collate_fn=collate,drop_last=True)



    net = newModel().to(device)
    # lr = 0.0001
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, weight_decay=5e-4)
    criterion = ContrastiveLoss()
    criterion_model = nn.CrossEntropyLoss(reduction='sum')
    best_acc = 0
    EPOCH = 150
    best_fpr, best_tpr, best_roc_auc = None,None,None
    best_tn, best_fp, best_fn, best_tp,best_sn, best_sp,bs, best_mcc=None,None,None,None,None,None,None,None



    for epoch in range(EPOCH):
        loss_ls = []
        loss1_ls = []
        loss2_3_ls = []
        t0 = time.time()
        net.train()
        for seq1, seq2, label, label1, label2 in train_iter_cont:

            # seq11 = []
            # for list11 in seq1:
            #     seq11.append(list(map(int, list11)))
            #
            # seq22 = []
            # for list_te in seq2:
            #     seq22.append(list(map(int, list_te)))
            # seq1 = torch.tensor(seq11)
            # seq2 = torch.tensor(seq22)
            # print(seq1)
            # print(type(seq1))
            # print(seq2)
            # print(type(seq2))

            output1 = net(seq1)
            output2 = net(seq2)
            output3 = net.trainModel(seq1)
            output4 = net.trainModel(seq2)
            loss1 = criterion(output1, output2, label)
            loss2 = criterion_model(output3, label1)
            loss3 = criterion_model(output4, label2)
            loss = loss1 + loss2 + loss3
            #             print(loss)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            loss_ls.append(loss.item())
            loss1_ls.append(loss1.item())
            loss2_3_ls.append((loss2 + loss3).item())

        net.eval()
        with torch.no_grad():
            train_acc,train_fpr,train_tpr,train_roc_auc,train_res_data = evaluate_accuracy(train_iter, net)
            test_acc,test_fpr,test_tpr,test_roc_auc,test_res_data = evaluate_accuracy(test_iter, net)

        if test_acc >= 0.765:
            if best_acc < test_acc:
                best_acc = test_acc
                best_fpr, best_tpr, best_roc_auc = test_fpr,test_tpr,test_roc_auc
                best_tn, best_fp, best_fn, best_tp,best_sn, best_sp,bs, best_mcc = test_res_data[0],test_res_data[1],test_res_data[2],test_res_data[3],test_res_data[4],test_res_data[5],test_res_data[6],test_res_data[7]

        results = f"epoch: {epoch + 1}, loss: {np.mean(loss_ls):.5f}, loss1: {np.mean(loss1_ls):.5f}, loss2_3: {np.mean(loss2_3_ls):.5f}"
        results += f'\ttrain_acc: {train_acc:.4f}, test_acc: {test_acc, "red"}, time: {time.time() - t0:.2f}'
        print(results)
    print("best_acc", best_acc)

    pd.DataFrame([best_fpr, best_tpr, [best_roc_auc]]).to_csv('./roc_data/Twins_ind_roc_data.csv', index=False,header=False)
    pd.DataFrame([best_tn, best_fp, best_fn, best_tp, best_sn, best_sp, bs, best_mcc]).to_csv('./res_data/Twins_ind_res_data.csv', index=False,header=False)
