import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader
from sklearn.preprocessing import normalize
from sklearn import metrics
import math
import numpy as np
from tqdm import tqdm
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, Coauthor
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv

from SelfSupervised_Representation_Augmentor.SEComm import Encoder, Model, drop_feature, SelfExpr, ClusterModel
from utils.eval import label_classification
from utils.sec_util import enhance_sim_matrix, post_proC, err_rate, best_map, load_wiki
from sklearn.decomposition import PCA
import os
from utils.ph_cc import ph_batch
from utils.early_stopper import EarlyStopper
import time
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    normalized_mutual_info_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)
from SelfSupervised_Representation_Augmentor.MGD import MGD
from utils.LOSS import Loss
from utils.MLP import MLPPredictor
from utils.funcs import *
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter
from SelfSupervised_Representation_Augmentor.SEComm import MergeLayer
from options import Args

writer=SummaryWriter("tsb_log")
args_set = Args()
args =  args_set.get_args()
LOG_FOLDER = "log/"+args.token+"_log"
date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
LOG_FOLDER += date
print(LOG_FOLDER)
os.makedirs(LOG_FOLDER, exist_ok=True)
LOG_PATH = os.path.join(LOG_FOLDER, "log.txt")
with open(LOG_PATH, "w") as f:
    f.write("start\n")
device = args_set.get_device()
data = args_set.get_dataset()
print("successfully loaded new data.")
data = data.to(device)
feature_extractor,neighbor_loader,memory,mergelayer,optimizer,semodel,\
    seoptimizer,clustermodel,clusteroptimizer,fulloptimizer=args_set.get_model()

min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())
train_loader = args_set.get_loader(data)
assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)
print("successfully built model.")
args_set.save_args(os.path.join(LOG_FOLDER, "args.txt"))


def test_spectral(c, y_train, n_class):
    y_train_x, _ = post_proC(c, n_class, 4, 1)
    print("Spectral Clustering Done.. Finding Best Fit..")
    scores = err_rate(y_train.detach().cpu().numpy(), y_train_x)
    return scores


def pretrain(epoch, save_epoch):
    memory.train()
    mergelayer.train()
    memory.reset_state()
    neighbor_loader.reset_state()
    total_loss = 0
    embeddings_train = []
    labels_train = []
    count = 0
    for batch in tqdm(train_loader):
        count +=1
        batch = batch.to(device)
        optimizer.zero_grad()
        src, dst, t, msg, label, attack = (
            batch.src,
            batch.dst,
            batch.t,
            batch.msg,
            batch.label,
            batch.attack,
        )
        n_id = torch.cat([src, dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)
        z, last_update = memory(n_id)
        ed, m = nodeMap(torch.stack((src, dst), dim=0))
        ed = ed.to(device)
        norm_factor, ed = cal_norm(ed, num_nodes=len(z), self_loop=False)
        z1 = mergelayer(drop_feature(z, args.drop_feature_rate_1))
        z2 = mergelayer(drop_feature(z, args.drop_feature_rate_2))
        if epoch == save_epoch:
            ph = ph_batch(src, dst, t, n_id).to(device)
            z_ph = torch.cat((z, ph), dim=1)
            embeddings_train.append(
                torch.cat([z_ph[assoc[src]], z_ph[assoc[dst]]], dim=1)
            )

            labels_train.append(batch.label)
        loss = mergelayer.loss(z1, z2, batch_size=args.batch_size)
        if args.use_mlp:
            msg_processed = feature_extractor(msg)
            memory.update_state(src, dst, t, msg_processed)
        else:
            memory.update_state(src, dst, t, msg)
        neighbor_loader.insert(src, dst)
        loss.backward()
        optimizer.step()
        memory.detach()
        total_loss += float(loss.item()) * batch.num_events
    if epoch == save_epoch:
        embeddings_save = torch.cat(embeddings_train, dim=0)
        torch.save(embeddings_save, "/new_disk/3D-IDS-Euler/edge_embeddings_train" + args.token + ".pt")
        labels = torch.cat(labels_train, dim=0)
        torch.save(labels, "/new_disk/3D-IDS-Euler/edge_labels_train" + args.token + ".pt")
    torch.cuda.empty_cache()
    return total_loss / data.num_events


@torch.no_grad()
def test(loader):
    torch.cuda.empty_cache()
    mergelayer.eval()
    torch.manual_seed(12345)
    preds, trues = [], []
    cutoff = 0.6
    for batch in loader:
        batch = batch.to(device)
        src, dst, t, msg, label, attack = (
            batch.src,
            batch.dst,
            batch.t,
            batch.msg,
            batch.label,
            batch.attack,
        )
        n_id = torch.cat([src, dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)
        ed, m = nodeMap(torch.stack((src, dst), dim=0))
        ed = ed.to(device)
        norm_factor, ed = cal_norm(ed, num_nodes=len(z), self_loop=False)
        z = mergelayer(z).to(device)
        score = torch.sum(z[assoc[src]] * z[assoc[dst]], dim=1)

        a = torch.where(
            score < cutoff,
            torch.tensor(0, device=device),
            torch.tensor(1, device=device),
        )
        preds += a.cpu()
        trues += label.cpu().tolist()
        preds += a.cpu()
        trues += label.cpu().tolist()
        memory.update_state(src, dst, t, msg)
        neighbor_loader.insert(src, dst)
    preds = np.array(preds).flatten()
    trues = np.array(trues).flatten()
    f1ss = f1_score(trues, preds, average="micro")
    return f1ss


def self_expressive_train(x_train, n_class):
    max_epoch = 20
    alpha = 5
    patience = 40
    x1 = x_train
    best_loss = 1e9
    bad_count = 0
    for epoch in range(max_epoch):
        semodel.train()
        seoptimizer.zero_grad()
        c, x2 = semodel(x1)
        se_loss = torch.norm(x1 - x2)
        reg_loss = torch.norm(c)
        loss = se_loss + alpha * reg_loss
        loss.backward()
        seoptimizer.step()
        print(
            "se_loss: {:.9f}".format(se_loss.item()),
            "reg_loss: {:.9f}".format(reg_loss.item()),
            end=" ",
        )
        print("full_loss: {:.9f}".format(loss.item()), flush=True)
        if loss.item() < best_loss:
            if torch.cuda.is_available():
                best_c = c.cpu()
            else:
                best_c = c
            bad_count = 0
            best_loss = loss.item()
        else:
            bad_count += 1
            if bad_count == patience:
                break

    C = best_c
    C = C.cpu().detach().numpy()
    L = enhance_sim_matrix(C, n_class, 4, 1)
    return L


def SE_layer(zs, ys):
    n_nodes = zs.shape[0]
    if args.norm:
        print("Normalizing embeddings before Self Expressive layer training")
        X = normalize(zs.detach().cpu().numpy())
        X = torch.tensor(X).to(device)
    else:
        X = zs
    from_list = []
    to_list = []
    val_list = []
    for iters in range(args.iterations):
        str = ""
        train_labels = random.sample(list(range(n_nodes)), args.batch_size)
        x_train = X[train_labels]
        y_train = ys[train_labels]
        print("\nStarting self expressive train iteration:", iters + 1)
        str += f"Starting self expressive train iteration:{iters+1:02d}\n"
        n_class = len(np.unique(y_train.detach().cpu().numpy()))
        S = self_expressive_train(x_train, n_class)

        print("Performance of Spectral Clustering on the Similarity Matrix")
        scores = test_spectral(S, y_train, n_class)
        print(" Ac:", scores[0], "NMI:", scores[1], "F1:", scores[2])
        str += f"Ac: {scores[0]:.4f}, NMI: {scores[1]:.4f},F1: {scores[2]:.4f}\n"
        print("\nRetriving similarity values for point pairs")
        with open(LOG_PATH, "a") as f:
            f.write(str)
        count = 0
        for i in range(args.batch_size):
            for j in range(args.batch_size):
                if i == j:
                    continue
                if S[i, j] >= (1 - args.threshold) or (
                    S[i, j] <= args.threshold and S[i, j] >= 0
                ):
                    from_list.append(train_labels[i])
                    to_list.append(train_labels[j])
                    val_list.append(S[i, j])
                    count += 1
        print("Included values for", count, "points out of", args.batch_size * args.batch_size)
        if scores[0] > args.SE_THRESHOLD:
            break
    print(len(from_list))
    np.save("/new_disk/3D-IDS-Euler/from_list" + args.token + ".npy", from_list)
    np.save("/new_disk/3D-IDS-Euler/to_list" + args.token + ".npy", to_list)
    np.save("/new_disk/3D-IDS-Euler/val_list" + args.token + ".npy", val_list)


alpha = 0.001
beta = 1


def train(from_list, to_list, val_list, num_events):
    mergelayer.train()
    clustermodel.train()
    memory.reset_state()
    neighbor_loader.reset_state()
    fulloptimizer.zero_grad()
    total_loss = 0
    embeddings_train = []
    for batch in tqdm(train_loader):
        batch = batch.to(device)
        src, dst, t, msg, label, attack = (
            batch.src,
            batch.dst,
            batch.t,
            batch.msg,
            batch.label,
            batch.attack,
        )
        n_id = torch.cat([src, dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)
        z, last_update = memory(n_id)
        ed, m = nodeMap(torch.stack((src, dst), dim=0))
        ed = ed.to(device)
        norm_factor, ed = cal_norm(ed, num_nodes=len(z), self_loop=False)
        z1 = mergelayer(drop_feature(z, args.drop_feature_rate_1))
        z2 = mergelayer(drop_feature(z, args.drop_feature_rate_2))
        embeddings_train.append(torch.cat([z[assoc[src]], z[assoc[dst]]], dim=1))
        loss = mergelayer.loss(z1, z2, batch_size=args.batch_size)
        if args.use_mlp:
            processed_msg=feature_extractor(msg)
            memory.update_state(src, dst, t, processed_msg)
        else:
            memory.update_state(src, dst, t, msg)
        neighbor_loader.insert(src, dst)
        memory.detach()
        total_loss += float(loss) * batch.num_events

    loss3 = total_loss / num_events
    z_full = clustermodel(torch.cat(embeddings_train, dim=0))
    z_from = z_full[from_list]
    z_to = z_full[to_list]
    pred_similarity = torch.sum(z_from * z_to, dim=1)
    print(pred_similarity.shape)
    print(val_list.shape)
    numer2 = torch.mm(z_full.T, z_full)
    denom2 = torch.norm(numer2)
    identity_mat = torch.eye(args.n_class)
    if torch.cuda.is_available():
        identity_mat = identity_mat.cuda()
    B = (identity_mat / math.sqrt(args.n_class)).to(device)
    C = (numer2 / denom2).to(device)
    loss1 = F.mse_loss(pred_similarity.to(device), torch.tensor(val_list).to(device))
    loss2 = torch.norm(B - C)
    loss = beta * loss3 + loss1 + alpha * loss2
    loss.backward()
    fulloptimizer.step()
    return loss, z_full


def online_pre(epoch, save_epoch, online_loader):
    memory.train()
    mergelayer.train()
    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.
    total_loss = 0
    embeddings_train = []
    labels_train = []
    count = 0
    for batch in tqdm(online_loader):
        count +=1
        batch = batch.to(device)
        optimizer.zero_grad()
        src, dst, t, msg, label, attack = (
            batch.src,
            batch.dst,
            batch.t,
            batch.msg,
            batch.label,
            batch.attack,
        )
        n_id = torch.cat([src, dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)
        z, last_update = memory(n_id)
        ed, m = nodeMap(torch.stack((src, dst), dim=0))
        ed = ed.to(device)
        norm_factor, ed = cal_norm(ed, num_nodes=len(z), self_loop=False)
        z1 = mergelayer(drop_feature(z, args.drop_feature_rate_1))
        z2 = mergelayer(drop_feature(z, args.drop_feature_rate_2))
        if epoch == save_epoch:
            ph = ph_batch(src, dst, t, n_id).to(device)
            z_ph = torch.cat((z, ph), dim=1)
            embeddings_train.append(
                torch.cat([z_ph[assoc[src]], z_ph[assoc[dst]]], dim=1)
            )

            labels_train.append(batch.label)
        loss = mergelayer.loss(z1, z2, batch_size=args.batch_size)
        if args.use_mlp:
            msg_processed = feature_extractor(msg)
            memory.update_state(src, dst, t, msg_processed)
        else:
            memory.update_state(src, dst, t, msg)
        neighbor_loader.insert(src, dst)
        loss.backward()
        optimizer.step()
        memory.detach()
        total_loss += float(loss.item()) * batch.num_events
    if epoch == save_epoch:
        embeddings_save = torch.cat(embeddings_train, dim=0)
        torch.save(embeddings_save, "/new_disk/3D-IDS-Euler/edge_embeddings" + args.token + ".pt")
        labels = torch.cat(labels_train, dim=0)
        torch.save(labels, "/new_disk/3D-IDS-Euler/edge_labels" + args.token + ".pt")
    torch.cuda.empty_cache()
    return total_loss / data.num_events

def get_online_data(data, online_ratio=0.2):
    indices = torch.randperm(len(data))
    online_size = int(len(data) * online_ratio)
    online_data = data[indices[:online_size]]
    return online_data


@torch.no_grad()
def online_evaluate(online_loader):
    mergelayer.eval()
    clustermodel.eval()
    memory.reset_state()
    neighbor_loader.reset_state()
    embeddings_online = []    
    for batch in tqdm(online_loader):
        batch = batch.to(device)
        src, dst, t, msg, label, attack = (
            batch.src,
            batch.dst,
            batch.t,
            batch.msg,
            batch.label,
            batch.attack,
        )
        n_id = torch.cat([src, dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)
        z, last_update = memory(n_id)
        embeddings_online.append(torch.cat([z[assoc[src]], z[assoc[dst]]], dim=1))
        memory.update_state(src, dst, t, msg)
        neighbor_loader.insert(src, dst)
        memory.detach()  
    z_full = clustermodel(torch.cat(embeddings_online, dim=0))
    identity_mat = torch.eye(args.n_class)
    if torch.cuda.is_available():
        identity_mat = identity_mat.cuda()

    
    ys = torch.load("/new_disk/3D-IDS-Euler/edge_labels" + args.token + ".pt")
    y_pred = torch.argmax(z_full, dim=1).cpu().detach().numpy()
    y_true = ys.detach().cpu().numpy()
    y_pred = best_map(y_true, y_pred)
    conf = confusion_matrix(y_true, y_pred)
    acc = metrics.accuracy_score(y_true, y_pred)
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    f1_macro = metrics.f1_score(y_true, y_pred, average="macro")
    f1_micro = metrics.f1_score(y_true, y_pred, average="micro")
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    fpr = conf[0][1] / (conf[0][1] + conf[1][1])
    
    
    y_true = label_binarize(y_true, classes=list(range(args.n_class)))
    y_pred = label_binarize(y_pred, classes=list(range(args.n_class)))
    auc = roc_auc_score(y_true, y_pred, average="micro")
    
    
    writer.add_scalar("Accuracy/online", acc)
    writer.add_scalar("F1-Macro/online", f1_macro)
    writer.add_scalar("F1-Micro/online", f1_micro)
    writer.add_scalar("Precision/online", precision)
    writer.add_scalar("Recall/online", recall)
    writer.add_scalar("FPR/online", fpr)
    writer.add_scalar("AUC/online", auc)

    print(f"Online Evaluation: "
        f"F1 Macro: {f1_macro:.4f}, F1 Micro: {f1_micro:.4f}, "
        f"AUC: {auc:.4f}, "
        f"FPR: {fpr:.4f}, ")

    with open(LOG_PATH, "a") as f:
        log_message = f"Online Evaluation: "
        log_message += f"F1 Macro: {f1_macro:.4f}, F1 Micro: {f1_micro:.4f}, "
        log_message += f"FPR: {fpr:.4f}, "
        f.write(log_message)



def main():
    with open(LOG_PATH, "a") as f:
        f.write("start pretrain()\n")
    for i in range(1, 21):
        log_str = ""
        loss_get = pretrain(i, 20)
        test_score = test(train_loader)
        log_str += f"Epoch: {i:02d}, Loss: {loss_get:.4f}\n"
        log_str += f"f1_score:{test_score:.4f}, Loss: {loss_get:.4f}\n"
        print(log_str)
        with open(LOG_PATH, "a") as f:
            f.write(log_str)
        if test_score > args.PRETRAIN_THRESHOLD:
            break
    torch.cuda.empty_cache()
    log_str = "successfully write zs and ys\n"
    zs = torch.load("/new_disk/3D-IDS-Euler/edge_embeddings_train" + args.token + ".pt")
    ys = torch.load("/new_disk/3D-IDS-Euler/edge_labels_train" + args.token + ".pt")
    print(zs.shape)
    print(ys.shape)
    ys_np = ys.cpu().numpy()
    with open(LOG_PATH, "a") as f:
        f.write(log_str)
    SE_layer(zs, ys)
    from_list = np.load("/new_disk/3D-IDS-Euler/from_list" + args.token + ".npy")
    val_list = np.load("/new_disk/3D-IDS-Euler/val_list" + args.token + ".npy")
    to_list = np.load("/new_disk/3D-IDS-Euler/to_list" + args.token + ".npy")
    with open(LOG_PATH, "a") as f:
        f.write("start train()")
    conf_list = []
    torch.cuda.empty_cache()
    for epoch in range(0, 100):
        torch.cuda.empty_cache()
        s = ""
        print("epoch: ", epoch)
        full_loss, z_full = train(from_list, to_list, val_list, data.num_events)
        print("loss: ", full_loss)
        s += f"Epoch: {epoch:02d}, Loss: {full_loss:.4f}\n"
        y_pred = torch.argmax(z_full, dim=1).cpu().detach().numpy()
        y_true = ys.detach().cpu().numpy()
        y_pred = best_map(y_true, y_pred)
        conf = confusion_matrix(y_true, y_pred)
        acc = metrics.accuracy_score(y_true, y_pred)
        nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
        f1_macro = metrics.f1_score(y_true, y_pred, average="macro")
        f1_micro = metrics.f1_score(y_true, y_pred, average="micro")
        precision = metrics.precision_score(y_true, y_pred)
        recall = metrics.recall_score(y_true, y_pred)
        fpr = conf[0][1] / (conf[0][1] + conf[1][1])
        y_true = label_binarize(y_true, classes=list(range(args.n_class)))
        y_pred = label_binarize(y_pred, classes=list(range(args.n_class)))
        auc = roc_auc_score(y_true, y_pred, average="micro")
        print("\n\nAc:", acc, "NMI:", nmi, "F1Ma:", f1_macro, "F1Mi:", f1_micro, "precision_score:", precision, "recall_score:", recall, "fpr:", fpr, "auc:", auc)
        s += f"Ac: {acc:.4f}, NMI: {nmi:.4f},F1Ma: {f1_macro:.4f},F1Mi: {f1_micro:.4f},precision_score: {precision:.4f},recall_score: {recall:.4f},fpr:{fpr:.4f}\n, auc:{auc:.4f}\n"
        writer.add_scalar("Loss/train", full_loss, epoch)
        writer.add_scalar("Accuracy/train", acc, epoch)
        writer.add_scalar("NMI/train", nmi, epoch)
        writer.add_scalar("F1-Macro/train", f1_macro, epoch)
        writer.add_scalar("F1-Micro/train", f1_micro, epoch)
        writer.add_scalar("Precision/train", precision, epoch)
        writer.add_scalar("Recall/train", recall, epoch)
        writer.add_scalar("FPR/train", fpr, epoch)
        writer.add_scalar("AUC/train", auc, epoch)
        print("confusion_matrix:\n", conf)
        s += "confusion_matrix" + np.array2string(conf) + "\n"
        with open(LOG_PATH, "a") as f:
            f.write(s)
        conf_list.append(conf)
    torch.save(conf_list, "conf" + args.token + ".pt")
    
    #online
    online_data = get_online_data(data, 0.2)
    print(f"Number:{len(online_data)}")
    online_loader = args_set.get_loader(online_data)
    online_pre(1, 1,  online_loader)
    online_evaluate(online_loader)

if __name__ == "__main__":
    main()
