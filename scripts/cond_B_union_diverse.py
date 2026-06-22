"""
조건 B 진짜 합집합 천장 + 효율 입증 (diverse)
==============================================
이전 B=241은 묶음발견 단순합(중복 포함). 진짜 천장은 중복 병합 후.
사용자 예측: B_union은 A(37)의 3배 이상이되 128 이내 수렴.
효율: 워드넷 기저가 천장 키워도 같은 예산서 변별ρ 더 높으면 효율적.

방법:
 1) supersense 묶음마다 발견(F1) → 각 발견자질의 '전체 토큰에 대한 이진 활성벡터' 보존.
    (묶음 안에서 z이진화했지만, 활성 토큰을 전체 N차원으로 되돌려 기록)
 2) 모은 자질들 간 코사인/자카드 중복 → 임계 이상이면 병합. 남은 수 = B_union.
 3) 효율: A자질집합 vs B_union자질집합, 자질수 늘리며 ρ(binaryJaccard vs embcos) 곡선 비교.
"""
import numpy as np, json, time
from scipy.stats import rankdata
from sklearn.cluster import KMeans

EMB=np.load('/mnt/user-data/uploads/news_diverse_embeddings.npy').astype(np.float64)
UNITS=json.load(open('/mnt/user-data/uploads/news_diverse_units.json'))
N=EMB.shape[0]
tok_ss=[u.get('ontological',[]) or [] for u in UNITS]
SS=sorted({x for p in tok_ss for x in p})

rng_pairs=np.random.default_rng(0)
PAIRS=rng_pairs.integers(0,N,size=(20000,2)); PAIRS=PAIRS[PAIRS[:,0]!=PAIRS[:,1]]
PI,PJ=PAIRS[:,0],PAIRS[:,1]
COS=np.einsum('ij,ij->i',EMB[PI],EMB[PJ])
CR=rankdata(COS); CR=CR-CR.mean(); CRN=CR/(np.linalg.norm(CR)+1e-12)

def rho_of_binary(Bcols):
    """Bcols: N x k 이진. ρ(Jaccard vs embcos)."""
    a=Bcols[PI]; b=Bcols[PJ]
    inter=(a&b).sum(1); union=(a|b).sum(1)
    with np.errstate(invalid='ignore',divide='ignore'):
        j=np.where(union>0,inter/union,0.0)
    if j.std()<1e-12: return 0.0
    jr=rankdata(j); jr=jr-jr.mean(); jn=jr/(np.linalg.norm(jr)+1e-12)
    return float(jn@CRN)

def make_candidates(emb_sub, idx_global, k_pool, z=1.5, seed=42):
    """묶음(또는 전체)에서 후보 이진자질 생성 → 전체 N차원 활성벡터 리스트."""
    n=emb_sub.shape[0]
    kp=min(k_pool, max(8, n//3))
    km=KMeans(n_clusters=kp,random_state=seed,n_init=3).fit(emb_sub)
    protos=km.cluster_centers_; protos/=(np.linalg.norm(protos,axis=1,keepdims=True)+1e-12)
    s=emb_sub@protos.T; Z=(s-s.mean(1,keepdims=True))/(s.std(1,keepdims=True)+1e-12)
    Bsub=(Z>z).astype(np.int8)   # n x kp
    # 전체 N차원으로 되돌림: 묶음 토큰만 활성, 나머지 0
    cols=[]
    for c in range(kp):
        if Bsub[:,c].sum()==0: continue
        full=np.zeros(N,dtype=np.int8)
        full[idx_global]=Bsub[:,c]
        cols.append(full)
    return cols   # 각 원소 = N차원 이진 활성

def greedy_select(cands, eps=0.002, max_feat=200):
    """후보 N차원 이진자질들에서 ρ 포화까지 greedy. 반환: 선택열 인덱스, ρ궤적."""
    sel=[]; cur=0.0; rem=list(range(len(cands)))
    traj=[]
    M=np.array(cands).T  # N x len
    while rem and len(sel)<max_feat:
        bc,br=None,cur
        for c in rem:
            r=rho_of_binary(M[:,sel+[c]])
            if r>br: br,bc=r,c
        if bc is None: break
        g=br-cur; sel.append(bc); rem.remove(bc); traj.append(br)
        if g<eps and len(sel)>=3: break
        cur=br
    return sel, traj, M

def merge_duplicates(M, cols, jacc_thresh=0.8):
    """자질 간 토큰활성 Jaccard ≥ thresh면 같은 자질로 병합. 남은 distinct 수."""
    K=M.shape[1]
    keep=[]; merged=0
    used=np.zeros(K,bool)
    for i in range(K):
        if used[i]: continue
        keep.append(i); used[i]=True
        ai=M[:,i].astype(bool)
        for jx in range(i+1,K):
            if used[jx]: continue
            aj=M[:,jx].astype(bool)
            inter=(ai&aj).sum(); union=(ai|aj).sum()
            if union>0 and inter/union>=jacc_thresh:
                used[jx]=True; merged+=1
    return len(keep), merged

SEEDS=[42,7,123]
t0=time.time()
print(f"diverse N={N}")
print("="*60)

# 묶음 분할(대표 supersense)
groups={s:[] for s in SS}
for i,parts in enumerate(tok_ss):
    if parts: groups[parts[0]].append(i)

resA=[]; resB_sum=[]; resB_union=[]; rhoA=[]; rhoB=[]
for seed in SEEDS:
    # 조건 A: 전체 발견
    candsA=make_candidates(EMB, np.arange(N), k_pool=200, seed=seed)
    selA,trajA,MA=greedy_select(candsA)
    resA.append(len(selA)); rhoA.append(trajA[-1] if trajA else 0)

    # 조건 B: 묶음별 발견 → 후보 다 모음
    candsB=[]
    for s in SS:
        idx=groups[s]
        if len(idx)<30: continue
        sub=EMB[np.array(idx)]
        cs=make_candidates(sub, np.array(idx), k_pool=200, seed=seed)
        # 묶음 안 greedy로 그 묶음 발견자질만 추림
        sel,_,Msub=greedy_select(cs)
        for k in sel: candsB.append(cs[k])
    resB_sum.append(len(candsB))
    MB=np.array(candsB).T
    # 진짜 합집합: 중복 병합
    uB,merged=merge_duplicates(MB, candsB, jacc_thresh=0.8)
    resB_union.append(uB)
    rhoB.append(rho_of_binary(MB))
    print(f"  seed={seed}: A={len(selA)} | B_sum={len(candsB)} merged={merged} "
          f"B_union={uB} | ρA={trajA[-1]:.3f} ρB={rho_of_binary(MB):.3f}  [{time.time()-t0:.0f}s]")

A=np.mean(resA); Bsum=np.mean(resB_sum); Buni=np.mean(resB_union)
print("\n"+"="*60)
print(f"천장 A          = {A:.1f}")
print(f"천장 B 단순합   = {Bsum:.1f}")
print(f"천장 B 합집합   = {Buni:.1f}  (중복병합 후 = 진짜 고유자질 수)")
print(f"B_union / A     = {Buni/A:.2f}배")
print(f"128 이내?       = {'예' if Buni<=128 else '아니오'}  (사용자 예측: 3배이상 & 128이내)")
print(f"\n변별 보존 ρ: A={np.mean(rhoA):.3f}  B(전체자질)={np.mean(rhoB):.3f}")
print(f"  → B의 ρ가 높으면: 천장 키운 대가로 변별 효율 삼 = 워드넷 기저 효율 입증")

json.dump(dict(A=float(A),B_sum=float(Bsum),B_union=float(Buni),
               ratio=float(Buni/A),within128=bool(Buni<=128),
               rhoA=float(np.mean(rhoA)),rhoB=float(np.mean(rhoB))),
          open('/home/claude/cond_B_union_diverse.json','w'),ensure_ascii=False,indent=2)
print("\n[saved] cond_B_union_diverse.json")
