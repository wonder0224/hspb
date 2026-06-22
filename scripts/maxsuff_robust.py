"""
의미 천장 robustness (508) — 천장 63이 설정에 흔들리나
======================================================
점검:
 ① 유의미 기준: A=최빈라벨 다름(기존) vs B=라벨Jaccard<0.5(더 엄격) vs C=Jaccard<0.7(느슨)
 ② 자질 순서: 활성수 큰 순(기존) vs 무작위 3회
 ③ 발견 시드: 42(기존) vs 7,123
천장 = 유의미이탈 이동평균(10)<0.5 첫 지점. 셋 다 비슷하면 robust.
"""
import numpy as np, json
from sklearn.cluster import KMeans

EMB=np.load('/mnt/user-data/uploads/news_legal_embeddings.npy').astype(np.float64)
UNITS=json.load(open('/mnt/user-data/uploads/news_legal_units.json'))
N=EMB.shape[0]
def labels_of(u):
    L=set()
    l1=u.get('l1','')
    for p in (l1.split('|') if l1 else []): L.add('ss:'+p)
    for ax in ['court','actor','action','topic']:
        v=u.get(ax,'')
        if v: L.add(f'{ax}:{v}')
    return L
TOK_LAB=[labels_of(u) for u in UNITS]

def cands_of(seed,k_pool=300,z=1.5):
    km=KMeans(n_clusters=k_pool,random_state=seed,n_init=3).fit(EMB)
    pr=km.cluster_centers_; pr/=(np.linalg.norm(pr,axis=1,keepdims=True)+1e-12)
    s=EMB@pr.T; Z=(s-s.mean(1,keepdims=True))/(s.std(1,keepdims=True)+1e-12)
    B=(Z>z).astype(np.int8)
    return [B[:,c] for c in range(k_pool) if B[:,c].sum()>0]

def labcount(grp):
    c={}
    for i in grp:
        for l in TOK_LAB[i]: c[l]=c.get(l,0)+1
    return c

def msplit(members,feat,crit):
    on=[i for i in members if feat[i]>0]; off=[i for i in members if feat[i]==0]
    if not on or not off: return 0,False
    moved=min(len(on),len(off))
    con,coff=labcount(on),labcount(off)
    if crit=='A':  # 최빈라벨 다름
        ton=max(con,key=con.get) if con else None
        toff=max(coff,key=coff.get) if coff else None
        sig=(ton!=toff and ton and toff)
    else:  # 라벨집합 Jaccard < thresh
        sa,sb=set(con),set(coff)
        u=len(sa|sb); jac=len(sa&sb)/u if u else 1.0
        thresh=0.5 if crit=='B' else 0.7
        sig=(jac<thresh)
    return moved,sig

def ceiling(cands,order,crit):
    arr=[[] for _ in range(N)]; sig_inc=[]
    for c in order:
        feat=cands[c]
        keys=[tuple(a) for a in arr]; classes={}
        for i,k in enumerate(keys): classes.setdefault(k,[]).append(i)
        ssig=0
        for k,m in classes.items():
            if len(m)<2: continue
            mv,sg=msplit(m,feat,crit)
            if sg: ssig+=mv
        for i in range(N): arr[i].append(int(feat[i]))
        sig_inc.append(ssig)
    sig=np.array(sig_inc)
    for s in range(len(sig)-10):
        if sig[s:s+10].mean()<0.5: return s+1
    return len(sig)

MAXF=120
print("508 robustness — 의미천장(기준값 63)")
print("="*56)

# ① 유의미 기준 (시드42, 활성수순 고정)
cands=cands_of(42)
order_act=sorted(range(len(cands)),key=lambda c:-int(cands[c].sum()))[:MAXF]
print("\n① 유의미 기준:")
for crit,name in [('A','최빈라벨다름(기존)'),('B','Jaccard<0.5(엄격)'),('C','Jaccard<0.7(느슨)')]:
    ce=ceiling(cands,order_act,crit)
    print(f"   {name:22s}: 천장 {ce}")

# ② 자질 순서 (시드42, 기준A 고정)
print("\n② 자질 순서 (기준A):")
print(f"   활성수큰순(기존)      : 천장 {ceiling(cands,order_act,'A')}")
rng=np.random.default_rng(1)
for t in range(3):
    od=list(range(len(cands))); rng.shuffle(od); od=od[:MAXF]
    print(f"   무작위{t+1}              : 천장 {ceiling(cands,od,'A')}")

# ③ 발견 시드 (활성수순, 기준A)
print("\n③ 발견 시드 (활성수순, 기준A):")
for sd in [42,7,123]:
    cs=cands_of(sd)
    od=sorted(range(len(cs)),key=lambda c:-int(cs[c].sum()))[:MAXF]
    print(f"   seed={sd:3d}              : 천장 {ceiling(cs,od,'A')}")

print("\n→ 셋 다 60~70 근처 유지면 천장 robust. 크게 벌어지면 설정 의존.")
