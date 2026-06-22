"""
사후해석 강건성 — 분기 집합의 자연부류성 (508)
================================================
질문: 자질배열로 분기된 토큰집합이 의미있는 자연부류를 구성하나.
구조적 사실: 자질 많을수록 집합 희소(수렴점=singleton). 자연부류 판정은 ≥3토큰만.
자연부류 점수 두 기준:
  (L) 라벨 기준 = 내적 라벨일관성 × 형제와 변별. 명명가능 자연부류.
  (E) 임베딩 기준 = 집합 내 평균 코사인 응집(라벨 안 씀). 잠재 자연부류.
대조군: 같은 크기 무작위 토큰집합. 분기 > 무작위면 분기가 자연부류 생성 입증.
깊이별: 얕은(큰집합)→깊은(희소) 자연부류성 추이.
"""
import numpy as np, json
from sklearn.cluster import KMeans
from collections import Counter

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

km=KMeans(n_clusters=200,random_state=42,n_init=3).fit(EMB)
pr=km.cluster_centers_; pr/=(np.linalg.norm(pr,axis=1,keepdims=True)+1e-12)
s=EMB@pr.T; Z=(s-s.mean(1,keepdims=True))/(s.std(1,keepdims=True)+1e-12)
Bd=(Z>1.5).astype(np.int8)
# 변별 순번
active=[c for c in range(200) if 0<int(Bd[:,c].sum())<N]
order=sorted(active,key=lambda c:-min(int(Bd[:,c].sum()),N-int(Bd[:,c].sum())))

def label_coherence(idx):
    """집합 내 라벨 일관성 = 평균 쌍 라벨 Jaccard."""
    if len(idx)<2: return None
    js=[]
    for a in range(len(idx)):
        for b in range(a+1,len(idx)):
            A,B=TOK_LAB[idx[a]],TOK_LAB[idx[b]]
            u=len(A|B); js.append(len(A&B)/u if u else 0)
    return float(np.mean(js))

def emb_cohesion(idx):
    if len(idx)<2: return None
    V=EMB[idx]; G=V@V.T
    iu=np.triu_indices(len(idx),1)
    return float(G[iu].mean())

def sibling_distinct(idx_a, idx_b):
    """두 형제집합 라벨 최빈이 다른가 + 라벨분포 거리."""
    def topl(idx):
        c=Counter()
        for i in idx:
            for l in TOK_LAB[i]: c[l]+=1
        return c
    ca,cb=topl(idx_a),topl(idx_b)
    if not ca or not cb: return 0
    ta=max(ca,key=ca.get); tb=max(cb,key=cb.get)
    return 1 if ta!=tb else 0

# 분기 트리: 자질 순번대로, 각 분기에서 자연부류 점수 기록(깊이별)
rng=np.random.default_rng(0)
def rand_baseline(size, metric, reps=20):
    vals=[]
    for _ in range(reps):
        idx=rng.choice(N,size=size,replace=False).tolist()
        v=metric(idx)
        if v is not None: vals.append(v)
    return np.mean(vals) if vals else None

records=[]  # (depth, size, lab_coh, emb_coh, sib_distinct, rand_lab, rand_emb)
def split(idx, depth, fpos):
    if fpos>=len(order) or len(idx)<3:
        return
    c=order[fpos]
    on=[i for i in idx if Bd[i,c]>0]; off=[i for i in idx if Bd[i,c]==0]
    if not on or not off:
        split(idx,depth,fpos+1); return
    # 이 분기의 두 자식 각각 자연부류성 (≥3만)
    for child,sib in [(on,off),(off,on)]:
        if len(child)>=3:
            lc=label_coherence(child); ec=emb_cohesion(child)
            sd=sibling_distinct(child,sib)
            rl=rand_baseline(len(child),label_coherence)
            re=rand_baseline(len(child),emb_cohesion)
            records.append((depth,len(child),lc,ec,sd,rl,re))
    split(on,depth+1,fpos+1); split(off,depth+1,fpos+1)

import sys; sys.setrecursionlimit(10000)
split(list(range(N)),0,0)
print(f"분기 자식집합(≥3토큰) {len(records)}개 기록")

# 깊이 구간별 집계
import numpy as np
recs=np.array([(d,sz,lc,ec,sd,rl,re) for (d,sz,lc,ec,sd,rl,re) in records
               if lc is not None and rl is not None],dtype=float)
print("\n깊이 | 집합수 | 평균크기 | 라벨일관(분기/무작위) | 임베딩응집(분기/무작위) | 형제변별율")
print("-"*90)
for lo,hi in [(0,2),(2,4),(4,6),(6,9),(9,20)]:
    m=(recs[:,0]>=lo)&(recs[:,0]<hi)
    if m.sum()==0: continue
    r=recs[m]
    print(f" {lo}-{hi-1:2d} | {int(m.sum()):5d}  | {r[:,1].mean():6.1f}  | "
          f"{r[:,2].mean():.3f} / {r[:,5].mean():.3f}      | "
          f"{r[:,3].mean():.3f} / {r[:,6].mean():.3f}       | {r[:,4].mean():.2f}")

# 전체 분기 vs 무작위 (자연부류성 우위?)
print("\n[전체] 분기집합 vs 무작위:")
print(f"  라벨일관:  분기 {recs[:,2].mean():.3f}  무작위 {recs[:,5].mean():.3f}  "
      f"향상 {recs[:,2].mean()-recs[:,5].mean():+.3f}")
print(f"  임베딩응집: 분기 {recs[:,3].mean():.3f}  무작위 {recs[:,6].mean():.3f}  "
      f"향상 {recs[:,3].mean()-recs[:,6].mean():+.3f}")
print(f"  형제변별율: {recs[:,4].mean():.2f} (형제와 최빈라벨 다른 비율)")
from scipy.stats import wilcoxon
try:
    _,p_lab=wilcoxon(recs[:,2],recs[:,5])
    _,p_emb=wilcoxon(recs[:,3],recs[:,6])
    print(f"  Wilcoxon p: 라벨 {p_lab:.2e}, 임베딩 {p_emb:.2e}")
except Exception as e:
    print("  wilcoxon:",e)

# 라벨로 잡히나 임베딩으로만 잡히나 (H3 연결)
lab_nat=(recs[:,2]>recs[:,5]+0.05).mean()
emb_nat=(recs[:,3]>recs[:,6]+0.02).mean()
print(f"\n  라벨기준 자연부류 비율: {lab_nat:.2f}")
print(f"  임베딩기준 자연부류 비율: {emb_nat:.2f}")
print(f"  → 임베딩>라벨이면 라벨로 못잡는 잠재 자연부류 존재(H3 연결)")

json.dump(dict(n=len(records),
               lab_branch=float(recs[:,2].mean()),lab_rand=float(recs[:,5].mean()),
               emb_branch=float(recs[:,3].mean()),emb_rand=float(recs[:,6].mean()),
               sib_distinct=float(recs[:,4].mean()),
               lab_nat_frac=float(lab_nat),emb_nat_frac=float(emb_nat)),
          open('/home/claude/naturalclass_508.json','w'))
print("\n[saved] naturalclass_508.json")
