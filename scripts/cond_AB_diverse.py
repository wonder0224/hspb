"""
조건 A vs B — 워드넷 기저가 발견 천장에 실제 영향 주나 (diverse, 4분류 세계)
============================================================================
사용자 의심: 워드넷 깖/안깖 천장 같음 = 인위적. (맞음 — 기존엔 워드넷이 발견에 안 들어감)
교정 실험:
  조건 A (워드넷 안 깖): 임베딩 전체에서 바로 F1 발견 → 천장 A.
  조건 B (워드넷 깖)   : supersense로 토큰 16묶음 → 각 묶음 안에서 따로 F1 발견
                         → 묶음별 발견폭 합산 = 천장 B.
판정:
  B≈A → 워드넷 잉여(임베딩이 이미 supersense 변별 포함). 천장일치=실질결과.
  B<A → 워드넷이 발견부담 덜어줌(경계 공짜제공). 진짜 부담분배.
  B>A → 워드넷 강제분할이 자연군집 가로질러 군집 더 필요. 워드넷 방해.
"""
import numpy as np, json, time
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from collections import Counter

EMB=np.load('/mnt/user-data/uploads/news_diverse_embeddings.npy').astype(np.float64)
UNITS=json.load(open('/mnt/user-data/uploads/news_diverse_units.json'))
N=EMB.shape[0]
SSF='ontological'
tok_ss=[u.get(SSF,[]) or [] for u in UNITS]
all_ss=[x for p in tok_ss for x in p]
SS=sorted(set(all_ss)); ss_idx={s:j for j,s in enumerate(SS)}

def discover(emb,k_pool=200,z=1.5,eps=0.002,max_pairs=20000,seed=42,min_n=30):
    """F1 발견. min_n 미만 토큰이면 발견 불가(None)."""
    n=emb.shape[0]
    if n<min_n: return None
    rng=np.random.default_rng(seed)
    kp=min(k_pool, max(8, n//3))   # 작은 묶음이면 후보풀 축소
    km=KMeans(n_clusters=kp,random_state=seed,n_init=3).fit(emb)
    protos=km.cluster_centers_; protos/=(np.linalg.norm(protos,axis=1,keepdims=True)+1e-12)
    s=emb@protos.T; Z=(s-s.mean(1,keepdims=True))/(s.std(1,keepdims=True)+1e-12)
    B=(Z>z).astype(np.int8)
    mp=min(max_pairs, n*(n-1)//2)
    pairs=rng.integers(0,n,size=(mp,2)); pairs=pairs[pairs[:,0]!=pairs[:,1]]
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
    sel=[]; cur=0.0; rem=[c for c in range(kp) if B[:,c].sum()>0]
    while rem:
        bc,br=None,cur
        for c in rem:
            r=rho(sel+[c])
            if r>br: br,bc=r,c
        if bc is None: break
        g=br-cur; sel.append(bc); rem.remove(bc)
        if g<eps and len(sel)>=3: break
        cur=br
    return len(sel)

SEEDS=[42,7,123]
t0=time.time()
print(f"diverse N={N}, supersense={len(SS)}")
print("="*60)

# --- 조건 A: 전체에서 발견 ---
print("조건 A (워드넷 안 깖): 전체 임베딩에서 F1 발견")
A_widths=[]
for seed in SEEDS:
    w=discover(EMB,seed=seed)
    A_widths.append(w)
    print(f"  seed={seed}: 천장 A = {w}  [{time.time()-t0:.0f}s]")
A=np.mean(A_widths)
print(f"  → 천장 A 평균 = {A:.1f}")

# --- 조건 B: supersense 묶음 안에서 각각 발견 → 합산 ---
print("\n조건 B (워드넷 깖): supersense 16묶음 안에서 각각 발견 → 합산")
# 토큰을 supersense에 귀속 (multi는 첫 supersense에 — 단순화, 묶음 분할용)
groups={s:[] for s in SS}
for i,parts in enumerate(tok_ss):
    if parts: groups[parts[0]].append(i)   # 대표 supersense

B_widths_per_seed=[]
for seed in SEEDS:
    total=0; detail=[]
    for s in SS:
        idx=groups[s]
        if len(idx)<30:
            detail.append((s,len(idx),'skip')); continue
        sub=EMB[idx]
        w=discover(sub,seed=seed)
        total+=(w or 0)
        detail.append((s,len(idx),w))
    B_widths_per_seed.append(total)
    if seed==SEEDS[0]:
        print(f"  [seed={seed}] 묶음별 발견폭:")
        for s,n,w in detail:
            print(f"     {s:22s} n={n:4d}  width={w}")
    print(f"  seed={seed}: 천장 B(합산) = {total}  [{time.time()-t0:.0f}s]")
Bsum=np.mean(B_widths_per_seed)
print(f"  → 천장 B 평균 = {Bsum:.1f}")

print("\n"+"="*60)
print("판정")
print("="*60)
print(f"  천장 A (워드넷 안 깖) = {A:.1f}")
print(f"  천장 B (워드넷 깖)    = {Bsum:.1f}")
print(f"  B/A = {Bsum/A:.2f}")
if Bsum < A*0.85:
    print("  → B<A: 워드넷이 발견부담 덜어줌(경계 공짜). 진짜 부담분배 측정됨.")
elif Bsum > A*1.15:
    print("  → B>A: 워드넷 강제분할이 자연군집 가로지름. 워드넷이 오히려 군집 더 요구.")
else:
    print("  → B≈A: 워드넷 잉여. 임베딩이 이미 supersense 변별 포함. 천장일치=실질결과.")

json.dump(dict(A=float(A),B=float(Bsum),ratio=float(Bsum/A),
               A_widths=A_widths,B_widths=B_widths_per_seed),
          open('/home/claude/cond_AB_diverse.json','w'),ensure_ascii=False,indent=2)
print("\n[saved] cond_AB_diverse.json")
