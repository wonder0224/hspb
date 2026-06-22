"""
(a) 분기 구조 검증
==================
가설: 발견 군집의 흡수(adj_purity)는 자질 크기(n_on)와 음의 상관.
      작고 특정적 자질 → 한 supersense로 흡수(갈래1).
      크고 바탕적 자질 → supersense 가로질러 직교(갈래2).
검증 1: n_on ↔ adj_purity 상관 (Spearman) + 다중 시드.
검증 2: supersense별 흡수 분해 — 어떤 고유범주가 군집과 정렬되나.
검증 3: 임계 무관 — adj_purity 분포 자체를 본다(단일 임계 안 씀).
"""
import numpy as np, json
from scipy.stats import spearmanr, rankdata
from sklearn.cluster import KMeans

EMB = np.load('/mnt/user-data/uploads/news_legal_embeddings.npy').astype(np.float64)
UNITS = json.load(open('/mnt/user-data/uploads/news_legal_units.json'))
N = EMB.shape[0]

# supersense 분할귀속 행렬
all_ss=[]; tok_ss=[]
for u in UNITS:
    parts = u['l1'].split('|') if u['l1'] else []
    tok_ss.append(parts); all_ss.extend(parts)
SS = sorted(set(all_ss)); ss_idx={s:j for j,s in enumerate(SS)}
S = np.zeros((N,len(SS)))
for i,parts in enumerate(tok_ss):
    if parts:
        w=1.0/len(parts)
        for p in parts: S[i,ss_idx[p]]+=w
base_rate = S.sum(axis=0)/N

def discover(emb, k_pool=200, z=1.5, eps=0.002, max_pairs=15000, seed=42):
    rng=np.random.default_rng(seed)
    n=emb.shape[0]
    km=KMeans(n_clusters=k_pool,random_state=seed,n_init=3).fit(emb)
    protos=km.cluster_centers_; protos/=(np.linalg.norm(protos,axis=1,keepdims=True)+1e-12)
    sim=emb@protos.T
    Z=(sim-sim.mean(1,keepdims=True))/(sim.std(1,keepdims=True)+1e-12)
    B=(Z>z).astype(np.int8)
    pairs=rng.integers(0,n,size=(max_pairs,2)); pairs=pairs[pairs[:,0]!=pairs[:,1]]
    pi,pj=pairs[:,0],pairs[:,1]
    cos=np.einsum('ij,ij->i',emb[pi],emb[pj])
    cr=rankdata(cos); cr=cr-cr.mean(); crn=cr/(np.linalg.norm(cr)+1e-12)
    def rho(cols):
        Bs=B[:,cols]; a=Bs[pi]; b=Bs[pj]
        inter=(a&b).sum(1); union=(a|b).sum(1)
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
    return sel, B[:,sel]

def feat_stats(B_sel):
    rows=[]
    for f in range(B_sel.shape[1]):
        on=B_sel[:,f].astype(bool)
        if on.sum()==0: continue
        mass=S[on].mean(0); top=int(mass.argmax())
        rows.append(dict(n_on=int(on.sum()), top_ss=SS[top],
                         raw=float(mass[top]), adj=float(mass[top]-base_rate[top])))
    return rows

# ---- 검증 1: 다중 시드에서 n_on ↔ adj 상관 ----
print("="*60)
print("검증 1: 자질 크기(n_on) ↔ 보정흡수(adj) 상관, 다중 시드")
print("="*60)
seed_corrs=[]
all_rows=[]
for seed in [42,7,123,2024,99]:
    sel,Bs=discover(EMB,seed=seed)
    rows=feat_stats(Bs)
    ns=[r['n_on'] for r in rows]; adj=[r['adj'] for r in rows]
    rho_corr=spearmanr(ns,adj).correlation
    seed_corrs.append(rho_corr)
    all_rows.extend(rows)
    print(f"  seed={seed:4d}: 발견폭={len(rows):2d}  corr(n_on,adj)={rho_corr:+.3f}")
print(f"\n  평균 상관: {np.mean(seed_corrs):+.3f}  (std {np.std(seed_corrs):.3f})")
print(f"  → 음수로 일관되면 '작을수록 흡수' 분기 확인")

# ---- 검증 2: supersense별 흡수 분해 (pooled) ----
print("\n"+"="*60)
print("검증 2: supersense별 흡수 분해 (5시드 풀)")
print("="*60)
from collections import defaultdict
by_ss=defaultdict(list)
for r in all_rows:
    by_ss[r['top_ss']].append(r['adj'])
print(f"  {'supersense':22s} {'기저율':>7s} {'#자질':>5s} {'평균adj':>8s} {'중앙n':>6s}")
ss_n = defaultdict(list)
for r in all_rows: ss_n[r['top_ss']].append(r['n_on'])
for s in sorted(by_ss, key=lambda x:-np.mean(by_ss[x])):
    print(f"  {s:22s} {base_rate[ss_idx[s]]:7.3f} {len(by_ss[s]):5d} "
          f"{np.mean(by_ss[s]):+8.3f} {int(np.median(ss_n[s])):6d}")

# ---- 검증 3: adj 분포 (임계 무관) ----
print("\n"+"="*60)
print("검증 3: adj_purity 분포 — 임계 안 쓰고 모양만")
print("="*60)
adj_all=np.array([r['adj'] for r in all_rows])
print(f"  n={len(adj_all)} 발견자질(5시드 풀)")
for p in [10,25,50,75,90]:
    print(f"  {p:2d}분위: {np.percentile(adj_all,p):+.3f}")
print(f"  평균 {adj_all.mean():+.3f}, std {adj_all.std():.3f}")
# 이봉성 단서: 작은자질 vs 큰자질 adj 분리
ns_all=np.array([r['n_on'] for r in all_rows])
med_n=np.median(ns_all)
small=adj_all[ns_all<=med_n]; large=adj_all[ns_all>med_n]
print(f"\n  작은자질(n<={med_n:.0f}): 평균adj {small.mean():+.3f} (n={len(small)})")
print(f"  큰자질  (n> {med_n:.0f}): 평균adj {large.mean():+.3f} (n={len(large)})")
print(f"  차이 {small.mean()-large.mean():+.3f}  → 양수면 분기 확인")
from scipy.stats import mannwhitneyu
u,pval=mannwhitneyu(small,large,alternative='greater')
print(f"  Mann-Whitney U(작>큰) p={pval:.4f}")

json.dump(dict(seed_corrs=seed_corrs, mean_corr=float(np.mean(seed_corrs)),
               by_ss={s:[round(float(np.mean(by_ss[s])),3),len(by_ss[s])] for s in by_ss},
               small_vs_large=[float(small.mean()),float(large.mean()),float(pval)]),
          open('/home/claude/g_branch_verify.json','w'), ensure_ascii=False, indent=2)
print("\n[saved] g_branch_verify.json")
