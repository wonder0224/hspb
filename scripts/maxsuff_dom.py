"""
최대-충분 포화 도메인 재현 — oncology / diverse
================================================
508에서 입증: 유의미 이탈이 ~63자질에서 0수렴(의미천장), ρ포화(~10)보다 늦지만 유한.
다른 도메인도 같으면 의미천장 유계수렴 도메인 독립.
라벨(유의미성 기준): 도메인별 supersense + 도메인 축.
  diverse: ontological(supersense) + domain
  onco   : l1(supersense) + pathology/part/domain
"""
import numpy as np, json, sys
from scipy.stats import rankdata
from sklearn.cluster import KMeans

def load(dom):
    if dom=='diverse':
        emb=np.load('/mnt/user-data/uploads/news_diverse_embeddings.npy')
        units=json.load(open('/mnt/user-data/uploads/news_diverse_units.json'))
        def lab(u):
            L=set()
            for p in (u.get('ontological',[]) or []): L.add('ss:'+p)
            for d in (u.get('domain',[]) or []): L.add('dom:'+d)
            return L
    elif dom=='onco':
        emb=np.load('/mnt/user-data/uploads/oncology_embeddings.npy')
        units=json.load(open('/mnt/user-data/uploads/oncology_units.json'))
        def lab(u):
            L=set()
            for p in (u.get('l1',[]) or []): L.add('ss:'+p)
            for ax in ['pathology','part','domain']:
                for v in (u.get(ax,[]) or []): L.add(f'{ax}:{v}')
            return L
    return emb.astype(np.float64), [lab(u) for u in units]

def run(dom, max_feat=120):
    EMB,TOK_LAB=load(dom)
    N=EMB.shape[0]
    km=KMeans(n_clusters=300,random_state=42,n_init=3).fit(EMB)
    pr=km.cluster_centers_; pr/=(np.linalg.norm(pr,axis=1,keepdims=True)+1e-12)
    s=EMB@pr.T; Z=(s-s.mean(1,keepdims=True))/(s.std(1,keepdims=True)+1e-12)
    B=(Z>1.5).astype(np.int8)
    cands=[B[:,c] for c in range(300) if B[:,c].sum()>0]
    order=sorted(range(len(cands)),key=lambda c:-int(cands[c].sum()))

    def msplit(members,feat):
        on=[i for i in members if feat[i]>0]; off=[i for i in members if feat[i]==0]
        if not on or not off: return 0,False
        moved=min(len(on),len(off))
        def top(grp):
            c={}
            for i in grp:
                for l in TOK_LAB[i]: c[l]=c.get(l,0)+1
            return max(c,key=c.get) if c else None
        ton,toff=top(on),top(off)
        return moved,(ton!=toff and ton is not None and toff is not None)

    arr=[[] for _ in range(N)]; sig_inc=[]
    for c in order[:max_feat]:
        feat=cands[c]
        keys=[tuple(a) for a in arr]; classes={}
        for i,k in enumerate(keys): classes.setdefault(k,[]).append(i)
        ssig=0
        for k,m in classes.items():
            if len(m)<2: continue
            mv,sg=msplit(m,feat)
            if sg: ssig+=mv
        for i in range(N): arr[i].append(int(feat[i]))
        sig_inc.append(ssig)
    sig=np.array(sig_inc)
    # 천장: 유의미이탈 이동평균<0.5 첫 지점
    ceil=None
    for s in range(len(sig)-10):
        if sig[s:s+10].mean()<0.5: ceil=s+1; break
    return N,sig,ceil

for dom in ['diverse','onco']:
    print("="*56); print(f"도메인 {dom}"); print("="*56)
    N,sig,ceil=run(dom)
    print(f"N={N}")
    print("구간별 평균 유의미이탈:")
    for s in range(0,len(sig),10):
        print(f"  step {s+1:3d}-{s+10:3d}: {sig[s:s+10].mean():.1f}")
    print(f"  → 유의미이탈 ≈0 수렴 ≈ step {ceil} (의미천장)")
    print(f"  → 508은 ~63. 유한 천장이면 도메인독립 유계수렴.\n")
    json.dump(dict(N=N,sig=sig.tolist(),ceiling=ceil),
              open(f'/home/claude/maxsuff_{dom}.json','w'))
print("[saved]")
