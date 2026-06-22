"""
LOC/EVENT 진단 — 왜 top-label로 안 뜨나
========================================
두 가설:
 (가) 표본부족: LOC 51, EVENT 34 → 군집을 못 채워 어떤 발견자질도 이들로 안 모임.
 (나) 흡수됨: LOC는 GPE에, EVENT는 다른 데로 빨려 독립 군집이 안 생김.
진단:
 1. LOC/EVENT 개체들이 임베딩 공간에서 GPE/그외와 얼마나 섞이나(최근접 라벨).
 2. 발견자질 중 LOC/EVENT를 '2순위'로 갖는 게 있나(top 아니어도 정렬).
 3. LOC 개체가 실제로 지명인지(NER 품질) 육안 확인.
"""
import numpy as np, json
from sklearn.cluster import KMeans
from scipy.stats import rankdata
from collections import Counter, defaultdict

EMB=np.load('/mnt/user-data/uploads/world_entities_emb.npy').astype(np.float64)
ENTS=json.load(open('/mnt/user-data/uploads/world_entities.json'))
N=EMB.shape[0]
LABELS=['PERSON','ORG','GPE','LOC','EVENT']
lab_idx={l:i for i,l in enumerate(LABELS)}
Y=np.zeros((N,len(LABELS)))
for i,e in enumerate(ENTS): Y[i,lab_idx[e['ner_label']]]=1.0
base_rate=Y.sum(0)/N
labels=[e['ner_label'] for e in ENTS]

# --- 진단 3: LOC/EVENT 개체 육안 ---
print("="*56); print("진단 3: LOC 개체 51개 (실제 지명인가?)"); print("="*56)
loc_ents=[e['text'] for e in ENTS if e['ner_label']=='LOC']
print(', '.join(loc_ents[:40]))
print("\nEVENT 개체 34개:")
ev_ents=[e['text'] for e in ENTS if e['ner_label']=='EVENT']
print(', '.join(ev_ents[:40]))

# --- 진단 1: k-NN 라벨 순도 (각 라벨 개체의 이웃이 같은 라벨인가) ---
print("\n"+"="*56); print("진단 1: 라벨별 k-NN 동질성 (k=10)"); print("="*56)
sim=EMB@EMB.T
np.fill_diagonal(sim,-1)
k=10
knn=np.argsort(-sim,axis=1)[:,:k]
print(f"  {'label':8s} {'#':>4s} {'同라벨이웃%':>10s} {'최다타라벨(혼동)':>20s}")
for l in LABELS:
    idx=[i for i in range(N) if labels[i]==l]
    if not idx: continue
    same=0; other=Counter()
    for i in idx:
        for j in knn[i]:
            if labels[j]==l: same+=1
            else: other[labels[j]]+=1
    frac=same/(len(idx)*k)
    conf=other.most_common(1)[0] if other else ('-',0)
    print(f"  {l:8s} {len(idx):4d} {frac*100:9.1f}% {conf[0]:>14s}({conf[1]})")

# --- 진단 2: 발견자질이 LOC/EVENT를 2순위로 갖나 ---
def discover(emb,k_pool=200,z=1.5,eps=0.002,max_pairs=20000,seed=42):
    rng=np.random.default_rng(seed); n=emb.shape[0]
    km=KMeans(n_clusters=k_pool,random_state=seed,n_init=3).fit(emb)
    protos=km.cluster_centers_; protos/=(np.linalg.norm(protos,axis=1,keepdims=True)+1e-12)
    s=emb@protos.T; Z=(s-s.mean(1,keepdims=True))/(s.std(1,keepdims=True)+1e-12)
    B=(Z>z).astype(np.int8)
    pairs=rng.integers(0,n,size=(max_pairs,2)); pairs=pairs[pairs[:,0]!=pairs[:,1]]
    pi,pj=pairs[:,0],pairs[:,1]
    cos=np.einsum('ij,ij->i',emb[pi],emb[pj])
    cr=rankdata(cos); cr=cr-cr.mean(); crn=cr/(np.linalg.norm(cr)+1e-12)
    def rho(cols):
        Bs=B[:,cols]; a=Bs[pi]; b=Bs[pj]
        inter=(a&b).sum(1); union=(a|b).sum(1)
        with np.errstate(invalid='ignore',divide='ignore'):
            j=np.where(union>0,inter/union,0.0)
        if j.std()<1e-12: return -1.0
        jr=rankdata(j); jr=jr-jr.mean(); jn=jr/(np.linalg.norm(jr)+1e-12)
        return float(jn@crn)
    sel=[]; cur=0.0; rem=[c for c in range(k_pool) if B[:,c].sum()>0]
    while rem:
        bc,br=None,cur
        for c in rem:
            r=rho(sel+[c])
            if r>br: br,bc=r,c
        if bc is None: break
        g=br-cur; sel.append(bc); rem.remove(bc)
        if g<eps and len(sel)>=3: break
        cur=br
    return B[:,sel]

print("\n"+"="*56); print("진단 2: 발견자질의 LOC/EVENT 정렬 (top 아니어도)"); print("="*56)
Bs=discover(EMB,seed=42)
loc_adj_as_top2=[]; ev_adj_as_top2=[]
loc_max=-9; ev_max=-9
for f in range(Bs.shape[1]):
    on=Bs[:,f].astype(bool)
    if on.sum()==0: continue
    mass=Y[on].mean(0)
    adj=mass-base_rate
    loc_max=max(loc_max, adj[lab_idx['LOC']])
    ev_max =max(ev_max,  adj[lab_idx['EVENT']])
print(f"  발견자질 전체에서 LOC adj 최댓값: {loc_max:+.3f} (어떤 자질도 LOC로 안 모이면 음수)")
print(f"  발견자질 전체에서 EVENT adj 최댓값: {ev_max:+.3f}")
print(f"  (기저율 LOC={base_rate[lab_idx['LOC']]:.3f}, EVENT={base_rate[lab_idx['EVENT']]:.3f})")
print(f"  → 최댓값도 0 근처면 표본부족(독립군집 없음), 양수 크면 일부 정렬")

json.dump(dict(loc_n=len(loc_ents),ev_n=len(ev_ents),
               loc_max_adj=float(loc_max),ev_max_adj=float(ev_max),
               loc_sample=loc_ents[:20]),
          open('/home/claude/h_loc_event_diag.json','w'),ensure_ascii=False,indent=2)
print("\n[saved] h_loc_event_diag.json")
