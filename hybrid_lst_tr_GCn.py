#!/usr/bin/env python
import os
import time
import math
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class ClinicalDataset(Dataset):
    def __init__(self, csv_path, sequence_length=5, normalization="standard"):
        self.sequence_length = sequence_length
        data = pd.read_csv(csv_path, delimiter=',')
        # drop non-numeric and mp4 columns
        exclude = {"Horodateur","Matricule du patient","Date du point hémodynamique","Heure du point hémodynamique",
                   "pulpe de l´index (main droite)","pulpe de l´index (main gauche)",
                   "l´éminence thénar  à droite","l´éminence thénar  à gauche"}
        data.drop(columns=[c for c in data.columns if c.strip() in exclude], errors='ignore', inplace=True)
        data = data.loc[:, ~data.columns.str.endswith('.mp4')]
        # convert to numeric, drop all-NaN
        valid = []
        for c in data.columns:
            data[c] = pd.to_numeric(data[c], errors='coerce')
            if data[c].notnull().any(): valid.append(c)
            else: logging.info(f"Dropping column '{c}' as non-numeric.")
        data = data[valid]
        data.fillna(0, inplace=True)
        # normalization
        if normalization == 'minmax':
            scaler = MinMaxScaler(); data.iloc[:, :] = scaler.fit_transform(data)
            logging.info("Applied MinMax normalization.")
        elif normalization == 'standard':
            scaler = StandardScaler(); data.iloc[:, :] = scaler.fit_transform(data)
            logging.info("Applied Standard normalization.")
        else:
            logging.info("No normalization applied.")
        self.data = data.values.astype(np.float32)
        self.num_features = self.data.shape[1]
        # build sequences
        N = len(self.data) - sequence_length
        if N <= 0:
            raise ValueError("Not enough data for given sequence_length")
        self.X = np.stack([self.data[i:i+sequence_length] for i in range(N)])
        self.y = np.stack([self.data[i+sequence_length] for i in range(N)])
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

class GraphModule(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.adj = nn.Parameter(torch.eye(input_dim))
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
    def forward(self, x):
        x = torch.matmul(x, self.adj)
        x = torch.relu(x)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class HybridModel(nn.Module):
    def __init__(self, input_dim, hidden_dim_lstm, num_layers_lstm,
                 dropout, transformer_layers, nhead, gnn_hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim_lstm, num_layers_lstm,
                            batch_first=True, dropout=dropout)
        encoder = nn.TransformerEncoderLayer(d_model=hidden_dim_lstm,
                                             nhead=nhead, dropout=dropout,
                                             batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder, num_layers=transformer_layers)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim_lstm, input_dim)
        self.gnn = GraphModule(input_dim, gnn_hidden_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        h, _ = self.lstm(x)                # (B, T, H)
        h = self.transformer(h)            # (B, T, H)
        h = self.avg_pool(h.transpose(1,2)).squeeze(-1)  # (B, H)
        h = self.dropout(h)
        out = self.fc(h)
        return self.gnn(out)

def compute_metrics(preds, targets, eps=1e-8):
    rmse = np.sqrt(np.mean((preds-targets)**2) + eps)
    mae  = np.mean(np.abs(preds-targets))
    mape = np.mean(np.abs((preds-targets)/(np.abs(targets)+eps))) * 100
    ss_res = np.sum((targets-preds)**2)
    ss_tot = np.sum((targets-np.mean(targets,0))**2) + eps
    r2 = 1 - ss_res/ss_tot
    acc = np.mean(np.abs(preds-targets)/(np.abs(targets)+eps) < 0.1)
    return rmse, mae, mape, r2, acc

def train_epoch(model, loader, crit, opt, device, clip):
    model.train(); total=0; loss_sum=0.0
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        opt.zero_grad()
        out = model(X)
        loss = crit(out,y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        loss_sum += loss.item()*X.size(0); total += X.size(0)
    return loss_sum/total

def validate_epoch(model, loader, crit, device):
    model.eval(); total=0; loss_sum=0.0; all_p=[]; all_t=[]
    with torch.no_grad():
        for X,y in loader:
            X,y = X.to(device), y.to(device)
            out= model(X)
            loss = crit(out,y)
            loss_sum += loss.item()*X.size(0); total += X.size(0)
            all_p.append(out.cpu().numpy()); all_t.append(y.cpu().numpy())
    preds = np.concatenate(all_p,0); targs = np.concatenate(all_t,0)
    metrics = compute_metrics(preds, targs)
    return loss_sum/total, metrics

def main():
    p = argparse.ArgumentParser()  
    p.add_argument("--csv_path", type=str, default="augmented_1M.csv")
    p.add_argument("--sequence_length", type=int, default=5)
    p.add_argument("--normalization", choices=["none","minmax","standard"], default="standard")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs",     type=int, default=10)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--dropout",    type=float, default=0.3)
    p.add_argument("--num_workers",type=int, default=8)
    p.add_argument("--checkpoint_dir", type=str, default="ckpts")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ClinicalDataset(args.csv_path, args.sequence_length, args.normalization)
    input_dim = dataset.num_features
    logging.info(f"Loaded dataset with {len(dataset)} samples, feature dim={input_dim}")

    hidden_dim = ((2*input_dim +31)//32)*32
    num_layers = min(4, max(2, hidden_dim//64))
    trans_layers = min(6, max(2, input_dim//10))
    nhead=1
    for h in (8,4,2,1):
        if hidden_dim % h==0: nhead=h; break
    gnn_hidden = input_dim
    clip_grad = math.ceil(math.sqrt(hidden_dim))

    logging.info("Auto‑tuned hyperparameters:")
    logging.info(f" hidden_dim_lstm     = {hidden_dim}")
    logging.info(f" num_layers_lstm    = {num_layers}")
    logging.info(f" transformer_layers = {trans_layers}")
    logging.info(f" nhead              = {nhead}")
    logging.info(f" gnn_hidden_dim     = {gnn_hidden}")
    logging.info(f" clip_grad          = {clip_grad}")

    # ----- Prepare DataLoaders -----
    train_n = int(0.8*len(dataset)); val_n = len(dataset)-train_n
    train_ds, val_ds = random_split(dataset, [train_n, val_n])
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers)

    model = HybridModel(
        input_dim=input_dim,
        hidden_dim_lstm=hidden_dim,
        num_layers_lstm=num_layers,
        dropout=args.dropout,
        transformer_layers=trans_layers,
        nhead=nhead,
        gnn_hidden_dim=gnn_hidden
    ).to(device)
    logging.info(model)

    crit = nn.MSELoss()
    opt  = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best = float('inf')
    for ep in range(1, args.epochs+1):
        t0=time.time()
        tr_loss = train_epoch(model, train_ld, crit, opt, device, clip_grad)
        val_loss, (rmse,mae,mape,r2,acc) = validate_epoch(model, val_ld, crit, device)
        dt = time.time()-t0
        logging.info(f"Epoch {ep}/{args.epochs} - Train:{tr_loss:.4f}  Val:{val_loss:.4f}  "
                     f"RMSE:{rmse:.4f} MAE:{mae:.4f} MAPE:{mape:.2f}% R2:{r2:.4f} Acc:{acc:.2f} Time:{dt:.1f}s")
        if val_loss<best:
            best=val_loss
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'best.pth'))
            logging.info(f"Saved best model (val {best:.4f})")

if __name__=='__main__': main()
