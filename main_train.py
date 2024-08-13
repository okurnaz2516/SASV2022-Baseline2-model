import os, sys, json, glob
import pickle as pkl
import shutil
import argparse
from utils import *
from models import *
from trainer import Trainer
from dataset import *
from torch.utils.data import DataLoader
from metrics import get_all_EERs_my


batch_size = 1024
lr = 0.001
epoch_size = 50
weight_decay = 1e-5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
trn_set = SASV_Dataset("trn")
trn_loader = DataLoader(trn_set, batch_size=batch_size, shuffle=True,
                        drop_last=False, pin_memory=True)

model = Baseline2()
model.to(device)
params = list(model.parameters())
optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
model.train()
min_eer = 1e4
for epoch in range(epoch_size):
    preds, keys = [], []
    trn_loss = 0
    tot_batch_trn = len(trn_loader)
    for num, data_minibatch in enumerate(trn_loader, 0):
        asv1, asv2, cm1, ans, key = data_minibatch
        with torch.set_grad_enabled(True):
            if torch.cuda.is_available():
                asv1 = asv1.to(device)
                asv2 = asv2.to(device)
                cm1 = cm1.to(device)
                ans = ans.to(device)

            pred = model(asv1, asv2, cm1)
            nloss = model.calc_loss(pred, ans)
            trn_loss += nloss
            optimizer.zero_grad()
            nloss.backward()
            optimizer.step()
            pred = torch.softmax(pred, dim=-1)
            preds.append(pred)
            keys.extend(list(key))

    preds = torch.cat(preds, dim=0)[:, 1].detach().cpu().numpy()

    trn_loss = (trn_loss/tot_batch_trn).item()
    sasv_eer_trn, sv_eer_trn, spf_eer_trn = get_all_EERs(
        preds=preds, keys=keys)

    print("\nEpoch-%d Trn: Loss: %0.5f, sasv_eer_trn: %0.3f, sv_eer_trn: %0.3f, spf_eer_trn: %0.3f" % (
        epoch+1,  trn_loss, 100 * sasv_eer_trn, 100 * sv_eer_trn, 100 * spf_eer_trn))

    model.eval()
    dev_set = SASV_Dataset("dev")
    dev_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=True,
                            drop_last=False, pin_memory=True)
    with torch.no_grad():
        preds, keys = [], []
        for num, data_minibatch in enumerate(dev_loader, 0):
            asv1, asv2, cm1, ans, key = data_minibatch
            if torch.cuda.is_available():
                asv1 = asv1.to(device)
                asv2 = asv2.to(device)
                cm1 = cm1.to(device)
                ans = ans.to(device)

            pred = model(asv1, asv2, cm1)
            nloss = model.calc_loss(pred, ans)
            pred = torch.softmax(pred, dim=-1)
            preds.append(pred)
            keys.extend(list(key))

        preds = torch.cat(preds, dim=0)[:, 1].detach().cpu().numpy()

        sasv_eer_dev, sv_eer_dev, spf_eer_dev = get_all_EERs(
            preds=preds, keys=keys)

        print("Epoch-%d Dev: sasv_eer_dev: %0.3f, sv_eer_dev: %0.3f, spf_eer_dev: %0.3f" % (
            epoch+1, 100 * sasv_eer_dev, 100 * sv_eer_dev, 100 * spf_eer_dev))

    if sasv_eer_dev < min_eer:
        torch.save(model.state_dict(), os.path.join(
            output_dir, "%s_best.pt" % (model.name)))
        min_eer = sasv_eer_dev
        best_epoch = epoch
        # print(f'Epoch-{epoch+1} Min sasv_eer: %{min_eer*100:.4f}')

print(
    f'\nMin sasv_eer_dev: %{min_eer*100:.4f} obtained in epoch {best_epoch+1}')


model.load_state_dict(torch.load(os.path.join(
    output_dir, "%s_best.pt" % (model.name))))
model.eval()
dev_set = SASV_Dataset("dev")
dev_loader = DataLoader(dev_set, batch_size=len(dev_set), shuffle=False,
                        drop_last=False, pin_memory=True)

with torch.no_grad():
    preds, keys = [], []
    for num, data_minibatch in enumerate(dev_loader, 0):
        asv1, asv2, cm1, ans, key = data_minibatch
        if torch.cuda.is_available():
            asv1 = asv1.to(device)
            asv2 = asv2.to(device)
            cm1 = cm1.to(device)
            ans = ans.to(device)

        pred = model(asv1, asv2, cm1)
        nloss = model.calc_loss(pred, ans)
        pred = torch.softmax(pred, dim=-1)
        preds.append(pred)
        keys.extend(list(key))

    preds = torch.cat(preds, dim=0)[:, 1].detach().cpu().numpy()

    sasv_eer_dev, sv_eer_dev, spf_eer_dev = get_all_EERs(
        preds=preds, keys=keys)
    print("\nEpoch-%d Dev: sasv_eer_dev: %0.3f, sv_eer_dev: %0.3f, spf_eer_dev: %0.3f" % (
        epoch+1, 100 * sasv_eer_dev, 100 * sv_eer_dev, 100 * spf_eer_dev))


eval_set = SASV_Dataset("eval")
eval_loader = DataLoader(eval_set, batch_size=len(eval_set), shuffle=False,
                          drop_last=False, pin_memory=True)

with torch.no_grad():
    preds, keys = [], []
    for num, data_minibatch in enumerate(eval_loader, 0):
        asv1, asv2, cm1, ans, key = data_minibatch
        if torch.cuda.is_available():
            asv1 = asv1.to(device)
            asv2 = asv2.to(device)
            cm1 = cm1.to(device)
            ans = ans.to(device)

        pred = model(asv1, asv2, cm1)
        nloss = model.calc_loss(pred, ans)
        pred = torch.softmax(pred, dim=-1)
        preds.append(pred)
        keys.extend(list(key))

    preds = torch.cat(preds, dim=0)[:, 1].detach().cpu().numpy()

    sasv_eer_eval, sv_eer_eval, spf_eer_eval = get_all_EERs(
        preds=preds, keys=keys)
    print("\nEpoch-%d Eval: sasv_eer_eval: %0.3f, sv_eer_eval: %0.3f, spf_eer_eval: %0.3f" % (
        epoch+1, 100 * sasv_eer_eval, 100 * sv_eer_eval, 100 * spf_eer_eval))
