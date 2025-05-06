import os
import random
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm

DATA_CSV      = 'data_vid.csv'
DATA_DIR      = 'downloaded_videos'
REGION_COLS   = [
    "pulpe de l´index (main droite)",
    "pulpe de l´index (main gauche)",
    "l´éminence thénar  à droite",
    "l´éminence thénar  à gauche",
]
LABEL_COL     = 'CRT1 (sec)'
PROCESSED_CSV = 'processed.csv'
SKELETON_DIR  = 'skeleton'
EPOCHS        = 50
BATCH_SIZE    = 32
CROP_LEN      = 64
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'

HAND_EDGES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]


PROTO_FILE    = 'models/pose_deploy.prototxt'
WEIGHTS_FILE  = 'models/pose_iter_102000.caffemodel'
MODEL_URLS = {
    PROTO_FILE:   "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/hand/pose_deploy.prototxt",
    WEIGHTS_FILE: "https://www.dropbox.com/s/6uzqorrexqpbi59/pose_iter_102000.caffemodel?dl=1"
}
IN_WIDTH, IN_HEIGHT = 368, 368
THRESH = 0.1


Path(PROTO_FILE).parent.mkdir(parents=True, exist_ok=True)
for path, url in MODEL_URLS.items():
    if not Path(path).exists():
        print(f"Downloading {path}...")
        urllib.request.urlretrieve(url, path)
net = cv2.dnn.readNetFromCaffe(PROTO_FILE, WEIGHTS_FILE)


def preprocess():
    """Flatten region‐video columns + CRT label into processed.csv."""
    df = pd.read_csv(DATA_CSV)
    df.columns = df.columns.str.strip(" ,")
    recs = []
    for _, row in df.iterrows():
        crt = row.get(LABEL_COL)
        if pd.isna(crt):
            continue
        for col in REGION_COLS:
            raw = row.get(col, "")
            if isinstance(raw, str):
                vid = raw.strip(" ,")
                if vid.lower().endswith(".mp4"):
                    recs.append({
                        'video_file': os.path.join(DATA_DIR, vid),
                        'crt': float(crt)
                    })
    if not recs:
        raise RuntimeError(f"No .mp4 entries found in columns: {REGION_COLS}")
    pd.DataFrame(recs).to_csv(PROCESSED_CSV, index=False)
    print(f"[OK] {len(recs)} entries saved to {PROCESSED_CSV}")


def extract_skeleton(path):
    """Extract (T,21,3) hand keypoints via OpenPose."""
    cap = cv2.VideoCapture(path)
    seq = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (IN_WIDTH, IN_HEIGHT),
                                     (0,0,0), swapRB=False, crop=False)
        net.setInput(blob)
        out = net.forward()  # (1,22,H_out,W_out)
        H_out, W_out = out.shape[2], out.shape[3]
        pts = []
        for i in range(22):
            pm = out[0, i, :, :]
            _, prob, _, pt = cv2.minMaxLoc(pm)
            x = (w * pt[0]) / W_out
            y = (h * pt[1]) / H_out
            pts.append([x, y, 0.0] if prob > THRESH else [0.0, 0.0, 0.0])
        seq.append(pts[:21])
    cap.release()
    return np.array(seq, dtype=np.float32)


class STGCNBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.gcn = nn.Conv2d(in_c, out_c, 1)

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, (9, 1), (stride, 1), (4, 0)),
            nn.BatchNorm2d(out_c)
        )

        self.down = nn.Identity() if (in_c == out_c and stride == 1) \
                    else nn.Conv2d(in_c, out_c, 1, (stride, 1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, Adj):
        Adj = Adj.to(x.device)
        # x: (N, C, T, V)
        xg = torch.einsum('nctv,vw->nctw', x, Adj)
        y = self.tcn(self.gcn(xg)) + self.down(x)
        return self.relu(y)


class STGCN_CRT(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(21 * 3)
        self.layers = nn.ModuleList([
            STGCNBlock(3, 64),
            STGCNBlock(64, 64),
            STGCNBlock(64, 64),
            STGCNBlock(64, 128, 2),
            STGCNBlock(128, 128),
            STGCNBlock(128, 256, 2),
            STGCNBlock(256, 256),
        ])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out = nn.Linear(256, 1)

    def forward(self, x):
        # x: (N, T, V=21, C=3)
        N, T, V, C = x.shape
        Adj = torch.zeros(V, V, device=x.device)
        for i, j in HAND_EDGES:
            Adj[i, j] = Adj[j, i] = 1

        # reshape to (N, C*V, T)
        x = x.permute(0, 3, 1, 2).reshape(N, C * V, T)
        x = self.bn(x).reshape(N, C, T, V)

        for layer in self.layers:
            x = layer(x, Adj)

        x = self.pool(x).flatten(1)
        return self.out(x).squeeze(-1)


class SkelDS(torch.utils.data.Dataset):
    def __init__(self, df, skel_dir, idxs):
        self.df = df.iloc[idxs].reset_index(drop=True)
        self.dir = Path(skel_dir)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        vid = row['video_file']
        p = self.dir / f"{Path(vid).stem}.npy"

        # load or re-extract, handling corrupted files
        if p.exists():
            try:
                sk = np.load(p)
                if sk.ndim != 3 or sk.shape[1:] != (21, 3):
                    raise ValueError
            except Exception:
                sk = extract_skeleton(vid)
        else:
            sk = extract_skeleton(vid)

        # fixed-length pad/crop
        if sk.shape[0] == 0:
            sk = np.zeros((CROP_LEN, 21, 3), dtype=np.float32)
        elif sk.shape[0] < CROP_LEN:
            pad = np.zeros((CROP_LEN, 21, 3), dtype=np.float32)
            pad[:sk.shape[0]] = sk
            sk = pad
        else:
            start = random.randint(0, sk.shape[0] - CROP_LEN)
            sk = sk[start:start + CROP_LEN]

        # cache cleaned skeleton
        p.parent.mkdir(exist_ok=True)
        np.save(p, sk)

        return torch.tensor(sk, dtype=torch.float32), torch.tensor(row['crt'], dtype=torch.float32)


def train_eval():
    df = pd.read_csv(PROCESSED_CSV)
    n = len(df)
    idxs = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(idxs)
    n0, n1 = int(0.7*n), int(0.85*n)
    splits = {
        'train': idxs[:n0],
        'val':   idxs[n0:n1],
        'test':  idxs[n1:]
    }

    loaders = {}
    for k, ids in splits.items():
        ds = SkelDS(df, SKELETON_DIR, ids)
        loaders[k] = torch.utils.data.DataLoader(
            ds, batch_size=BATCH_SIZE,
            shuffle=(k == 'train'),
            num_workers=4
        )

    model = STGCN_CRT().to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_f = nn.L1Loss()
    best_mae = float('inf')

    for e in range(1, EPOCHS + 1):
        model.train()
        tr_losses = []
        for x, y in tqdm(loaders['train'], desc=f"Epoch {e} train"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optim.zero_grad()
            p = model(x)
            l = loss_f(p, y)
            l.backward()
            optim.step()
            tr_losses.append(l.item())
        avg_tr = float(np.mean(tr_losses))

        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for x, y in tqdm(loaders['val'], desc=f"Epoch {e} val"):
                x, y = x.to(DEVICE), y.to(DEVICE)
                p = model(x).cpu().numpy()
                t = y.cpu().numpy()
                vp.append(p)
                vt.append(t)
        vp = np.concatenate(vp)
        vt = np.concatenate(vt)
        val_mae = mean_absolute_error(vt, vp)
        val_r2 = r2_score(vt, vp)
        print(f"E{e} train_L1={avg_tr:.3f} val_MAE={val_mae:.3f} val_R2={val_r2:.3f}")

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), 'best_model.pt')

    # Final test
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()
    tp, tt = [], []
    with torch.no_grad():
        for x, y in tqdm(loaders['test'], desc="Test set"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            p = model(x).cpu().numpy()
            t = y.cpu().numpy()
            tp.append(p)
            tt.append(t)
    tp = np.concatenate(tp)
    tt = np.concatenate(tt)
    print(f"Test MAE={mean_absolute_error(tt, tp):.3f} R2={r2_score(tt, tp):.3f}")
    torch.save(model.state_dict(), 'final_model.pt')
    print("Saved best_model.pt & final_model.pt")


if __name__ == '__main__':
    preprocess()
    train_eval()
