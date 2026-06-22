"""
입도 스윕 (508 legal) — H4 직접 시험
=====================================
같은 발견 군집(F1)을 세 입도에서 흡수율 측정:
  거침(coarse): supersense 16 → 상위묶음 4
  중간(mid)   : supersense 16 그대로
  잘음(fine)  : 4축 라벨(court/actor/action/topic) 펼침
H4 예측: 발견은 임베딩이 실제 쓰는 입도를 따른다.
  → 흡수율이 어느 입도에서 최대인가? 양끝(너무 거침/너무 잘음)에서 떨어지나?
주의: 입도마다 라벨 수·기저율이 다르므로 보정 purity(우연 대비)로 비교해야 공정.
"""
import numpy as np, json
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from collections import defaultdict

EMB=np.load('/mnt/user-data/uploads/news_legal_embeddings.npy').astype(np.float64)
UNITS=json.load(open('/mnt/user-data/uploads/news_legal_units.json'))
N=EMB.shape[0]

# ---- 입도별 라벨 귀속행렬 만들기 ----
# 중간: supersense (분할귀속)
all_ss=[]; tok_ss=[]
for u in UNITS:
    parts=u['l1'].split('|') if u['l1'] else []
    tok_ss.append(parts); all_ss.extend(parts)
SS=sorted(set(all_ss)); ss_idx={s:j for j,s in enumerate(SS)}

# 거침: supersense → 4 상위묶음 (결정적 규칙)
COARSE_MAP={
 'noun.person':'ENTITY','noun.group':'ENTITY','noun.location':'ENTITY',
 'noun.object':'ENTITY','noun.artifact':'ENTITY','noun.substance':'ENTITY','noun.body':'ENTITY',
 'noun.event':'EVENTIVE','noun.act':'EVENTIVE','noun.state':'EVENTIVE',
 'noun.process':'EVENTIVE','noun.phenomenon':'EVENTIVE',
 'noun.communication':'INFO','noun.cognition':'INFO','noun.attribute':'INFO',
 'noun.time':'OTHER',
}
COARSE=sorted(set(COARSE_MAP.values()))
c_idx={c:j for j,c in enumerate(COARSE)}

# 잘음: 4축 라벨 전체 펼침 (court/actor/action/topic 각 값이 개별 라벨)
AXES=['court','actor','action','topic']
fine_labels=set()
for u in UNITS:
    for ax in AXES:
        v=u.get(ax,'')
        if v: fine_labels.add(f"{ax}:{v}")
FINE=sorted(fine_labels); f_idx={l:j for j,l in enumerate(FINE)}

def build_Y(level):
    if level=='mid':
        Y=np.zeros((N,len(SS)))
        for i,parts in enumerate(tok_ss):
            if parts:
                w=1.0/len(parts)
                for p in parts: Y[i,ss_idx[p]]+=w
        return Y,SS
    if level=='coarse':
        Y=np.zeros((N,len(COARSE)))
        for i,parts in enumerate(tok_ss):
            if parts:
                w=1.0/len(parts)
                for p in parts: Y[i,c_idx[COARSE_MAP[p]]]+=w
        return Y,COARSE
    if level=='fine':
        Y=np.zeros((N,len(FINE)))
        for i,u in enumerate(UNITS):
            labs=[f"{ax}:{u[ax]}" for ax in AXES if u.get(ax,'')]
            if labs:
                w=1.0/len(labs)
                for l in labs: Y[i,f_idx[l]]+=w
            # 4축 빈 토큰은 행 0 (어디에도 흡수 안 됨 — 정직)
        return Y,FINE

def discover(emb,k_pool=200,z=1.5,eps=0.002,max_pairs=15000,seed=42):
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

def absorb(B_sel,Y,base_rate):
    """발견자질별 최빈라벨 보정purity. 4축빈토큰(행0)은 해당 자질에서 제외 안 하고
    그대로 — mass가 0이면 흡수 0으로 잡힘(정직)."""
    adjs=[]; raws=[]
    for f in range(B_sel.shape[1]):
        on=B_sel[:,f].astype(bool)
        if on.sum()==0: continue
        mass=Y[on].mean(0)
        if mass.sum()<1e-9:   # 켠 토큰들이 이 입도 라벨이 전혀 없음
            adjs.append(0.0); raws.append(0.0); continue
        top=int(mass.argmax())
        raws.append(float(mass[top]))
        adjs.append(float(mass[top]-base_rate[top]))
    return np.array(adjs),np.array(raws)

print("입도별 라벨 수:")
for lv in ['coarse','mid','fine']:
    Y,names=build_Y(lv)
    print(f"  {lv:7s}: {len(names)}개 라벨  (커버리지: 라벨있는토큰 {int((Y.sum(1)>0).sum())}/{N})")

print("\n"+"="*60)
print("입도 스윕 — 같은 발견군집, 세 입도 흡수율 (5시드 풀)")
print("="*60)
results={}
for lv in ['coarse','mid','fine']:
    Y,names=build_Y(lv)
    base=Y.sum(0)/N
    all_adj=[]; all_raw=[]
    for seed in [42,7,123,2024,99]:
        Bs=discover(EMB,seed=seed)
        a,r=absorb(Bs,Y,base)
        all_adj.extend(a); all_raw.extend(r)
    all_adj=np.array(all_adj); all_raw=np.array(all_raw)
    results[lv]={'n_labels':len(names),'mean_adj':float(all_adj.mean()),
                 'mean_raw':float(all_raw.mean()),'median_adj':float(np.median(all_adj))}
    print(f"  {lv:7s} ({len(names):2d}라벨): 평균 보정adj {all_adj.mean():+.3f}  "
          f"평균 raw {all_raw.mean():.3f}  중앙adj {np.median(all_adj):+.3f}")

print("\n[해석]")
adj_c,adj_m,adj_f=results['coarse']['mean_adj'],results['mid']['mean_adj'],results['fine']['mean_adj']
print(f"  거침 {adj_c:+.3f} → 중간 {adj_m:+.3f} → 잘음 {adj_f:+.3f}")
if adj_m>=adj_c and adj_m>=adj_f:
    print("  → 중간(supersense)에서 최대. 발견이 거기 입도에 정렬 = H4 지지(임베딩 입도 따름).")
elif adj_f>adj_m:
    print("  → 잘음(4축)에서 최대. 발견이 도메인 입도까지 따라감 = 입도 더 잘게 가능.")
else:
    print("  → 거침에서 최대. 발견이 굵은 범주만 잡음.")

json.dump(results,open('/home/claude/granularity_sweep_508.json','w'),ensure_ascii=False,indent=2)
print("\n[saved] granularity_sweep_508.json")
