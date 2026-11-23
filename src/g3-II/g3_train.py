"""G3-II (Kamatar et al. 2025 style) fine-tuning on the shared Chow 68-class split.

Run from the repo root:
  python src/g3-II/g3_train.py --model densenet169 --seed 0 --val_fold 0
  python src/g3-II/g3_train.py --model convnext_small --seed 0 --val_fold 0 --cbam
Smoke test: add  --epochs_a 1 --epochs_b 1
Needs: splits/chow68.csv (path,class,split,fold,dup_group), splits/classes.txt, images under --img_root
Writes: results/preds/g3ii_<model>[_cbam]_rgb_s<seed>.csv  (path,true,where,<68 probs>)
        models/g3ii_<model>[_cbam]_rgb_s<seed>.pt          (git-ignored)
Test rows are scored ONCE, at the end, with the model chosen on the validation fold.
"""
import argparse, os, random, copy
import numpy as np, pandas as pd, torch, torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms as T

p = argparse.ArgumentParser()
p.add_argument('--model', required=True, choices=['densenet169', 'densenet201', 'convnext_small', 'mobilenet_v2'])
p.add_argument('--seed', type=int, default=0)
p.add_argument('--val_fold', type=int, default=0)
p.add_argument('--cbam', action='store_true')
p.add_argument('--img_root', default='data/images')
p.add_argument('--bs', type=int, default=32)
p.add_argument('--epochs_a', type=int, default=5)
p.add_argument('--epochs_b', type=int, default=25)
p.add_argument('--patience', type=int, default=5)
a = p.parse_args()

random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
classes = [c.strip() for c in open('splits/classes.txt', encoding='utf-8') if c.strip()]
cid = {c: i for i, c in enumerate(classes)}
df = pd.read_csv('splits/chow68.csv')
df['path'] = df['path'].str.replace('Gem Stone Images Dataset/', '', regex=False)
missing = [p_ for p_ in df['path'] if not os.path.exists(os.path.join(a.img_root, p_))]
assert not missing, f'{len(missing)} images not found under {a.img_root}, e.g. {missing[:3]}'
assert set(df['class']) <= set(classes), f"classes in chow68.csv missing from classes.txt: {set(df['class']) - set(classes)}"
test = df[df.split == 'test']
trainall = df[df.split == 'train']
val = trainall[trainall.fold == a.val_fold]
train = trainall[trainall.fold != a.val_fold]
print(f'device {dev} | train {len(train)} | val {len(val)} | test {len(test)}')

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
tf_train = T.Compose([T.RandomResizedCrop(224, scale=(0.6, 1.0)), T.RandomHorizontalFlip(), T.RandomRotation(15),
                      T.ColorJitter(brightness=0.2, contrast=0.2),  # no hue/saturation jitter: colour is the cue
                      T.ToTensor(), T.Normalize(MEAN, STD)])
tf_eval = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])

class DS(Dataset):
    def __init__(s, d, tf): s.d, s.tf = d.reset_index(drop=True), tf
    def __len__(s): return len(s.d)
    def __getitem__(s, i):
        r = s.d.iloc[i]
        img = Image.open(os.path.join(a.img_root, r.path)).convert('RGB')
        return s.tf(img), cid[r['class']]

NW = 0 if os.name == 'nt' else 2   # Windows: worker processes re-import the script
def loader(d, tf, shuffle):
    return DataLoader(DS(d, tf), batch_size=a.bs, shuffle=shuffle, num_workers=NW, pin_memory=(dev == 'cuda'))

class CBAM(nn.Module):  # Woo et al. 2018: channel attention, then spatial attention
    def __init__(s, c, r=16):
        super().__init__()
        s.mlp = nn.Sequential(nn.Linear(c, c // r), nn.ReLU(), nn.Linear(c // r, c))
        s.sp = nn.Conv2d(2, 1, 7, padding=3)
    def forward(s, x):
        w = torch.sigmoid(s.mlp(x.mean((2, 3))) + s.mlp(x.amax((2, 3))))[:, :, None, None]
        x = x * w
        m = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        return x * torch.sigmoid(s.sp(m))

def build():
    n = len(classes)
    m = getattr(models, a.model)(weights='DEFAULT')
    if a.model.startswith('densenet'):
        m.classifier = nn.Linear(m.classifier.in_features, n)
        head = m.classifier
        last = [m.features.denseblock4, m.features.norm5]
    elif a.model == 'convnext_small':
        last = list(m.features)[-3:]                 # taken BEFORE CBAM is added, so both variants unfreeze the same layers
        cb = CBAM(768) if a.cbam else None
        if cb is not None:
            m.features = nn.Sequential(*m.features, cb)
        m.classifier[2] = nn.Linear(768, n)
        head = nn.ModuleList([m.classifier[2]] + ([cb] if cb is not None else []))  # CBAM trains in both stages
    else:
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n)
        head = m.classifier
        last = list(m.features)[-3:]
    return m.to(dev), head, last

def opt_for(m, head, last, stage):
    for q in m.parameters(): q.requires_grad = False
    for q in head.parameters(): q.requires_grad = True
    head_lr = 1e-4 if a.model == 'mobilenet_v2' else 1e-2
    bb_lr = 1e-4 if a.model == 'mobilenet_v2' else 1e-3
    if stage == 'A':
        groups = [{'params': list(head.parameters()), 'lr': head_lr}]
    else:
        bb = [q for blk in last for q in blk.parameters()]
        for q in bb: q.requires_grad = True
        groups = [{'params': list(head.parameters()), 'lr': head_lr}, {'params': bb, 'lr': bb_lr}]
    if a.model == 'mobilenet_v2':
        return torch.optim.RMSprop(groups, weight_decay=1e-4)
    return torch.optim.SGD(groups, momentum=0.9, weight_decay=1e-4)

@torch.no_grad()
def probs(m, d):
    m.eval(); out = []
    for x, _ in loader(d, tf_eval, False): out.append(torch.softmax(m(x.to(dev)), 1).cpu())
    return torch.cat(out).numpy()

def run_epochs(m, o, n, best, tag):
    crit = nn.CrossEntropyLoss(); bad = 0
    for ep in range(n):
        m.train()
        for x, y in loader(train, tf_train, True):
            o.zero_grad(); crit(m(x.to(dev)), y.to(dev)).backward(); o.step()
        pv = probs(m, val); yv = np.array([cid[c] for c in val['class']])
        vl = -np.log(pv[np.arange(len(yv)), yv] + 1e-9).mean(); acc = (pv.argmax(1) == yv).mean()
        print(f'{tag} ep{ep} val_loss {vl:.4f} val_acc {acc:.4f}', flush=True)
        if vl < best['loss']:
            best.update(loss=vl, state=copy.deepcopy(m.state_dict())); bad = 0
        else:
            bad += 1
            if tag == 'B' and bad >= a.patience: break

model, head, last = build()
best = {'loss': 1e9, 'state': None}
run_epochs(model, opt_for(model, head, last, 'A'), a.epochs_a, best, 'A')   # stage A: new head (and CBAM) only
model.load_state_dict(best['state'])
run_epochs(model, opt_for(model, head, last, 'B'), a.epochs_b, best, 'B')   # stage B: unfreeze deeper layers
model.load_state_dict(best['state'])

name = f"g3ii_{a.model}{'_cbam' if a.cbam else ''}_rgb_s{a.seed}"
os.makedirs('results/preds', exist_ok=True)
os.makedirs('models', exist_ok=True)
rows = []
for where, d in (('oof', val), ('test', test)):
    pr = probs(model, d)
    t = pd.DataFrame(pr, columns=classes); t.insert(0, 'where', where)
    t.insert(0, 'true', d['class'].values); t.insert(0, 'path', d['path'].values); rows.append(t)
pd.concat(rows).to_csv(f'results/preds/{name}.csv', index=False)
torch.save(model.state_dict(), f'models/{name}.pt')
print('saved', name)