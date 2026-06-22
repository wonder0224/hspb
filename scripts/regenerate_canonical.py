"""
regenerate_canonical.py  —  [당신 PC: C:\CL_token_experiment 에서 실행]
======================================================================
§5 정본 재생성. 한 시드세트(5개)로 P1/P3/P4/N 전부 측정 → 단일 JSON.
새 채팅에 canonical_results.json 하나만 올리면 §5 수치 전부 정합.

P1 의미천장(J)   : 발견자질 끝까지 추가, 유의미 이탈 0수렴 깊이.
P3 sublinear(O)  : N부분표본 포화k → c=k/log2N, log/sqrt/linear 피팅. 5데이터셋.
P4 자연부류(M)   : 분기집합 vs 무작위 라벨/임베딩 응집.
N  robustness    : z×k_pool sublinear 유지 (onco).
모두 seed∈{42,7,123,2024,99} 각각 → 평균±std.

준비: pip install numpy scipy scikit-learn
실행: python regenerate_canonical.py
출력: C:\CL_token_experiment\canonical_results.json
시간: PC에 따라 30분~1.5시간 (5시드×5데이터셋×여러측정). 진행상황 출력됨.
"""
import numpy as np, json, time, os
from scipy.stats import rankdata
from sklearn.cluster import KMeans

BASE = r"C:\CL_token_experiment"
SEEDS = [42, 7, 123, 2024, 99]
OUT = os.path.join(BASE, "canonical_results.json")

# ---- 데이터셋 로더 (필드차 흡수) ----
def load(key):
    d=os.path.join(BASE,"data")
    if key=='508-OpenAI':
        E=np.load(d+r"\508\news_legal_embeddings.npy")
        U=json.load(open(d+r"\508\news_legal_units.json",encoding="utf-8"))
        ss=lambda u:(u.get('l1','').split('|') if u.get('l1','') else [])
        axes=['court','actor','action','topic']; axget=lambda u,a:( [u[a]] if u.get(a,'') else [])
    elif key=='diverse-OpenAI':
        E=np.load(d+r"\diverse\news_diverse_embeddings.npy")
        U=json.load(open(d+r"\diverse\news_diverse_units.json",encoding="utf-8"))
        ss=lambda u:(u.get('ontological',[]) or [])
        axes=['domain']; axget=lambda u,a:(u.get(a,[]) or [])
    elif key.startswith('onco'):
        emb={'onco-OpenAI':r"\oncology\oncology_embeddings.npy",
             'onco-E5':r"\oncology\oncology_emb_e5.npy",
             'onco-MPNet':r"\oncology\oncology_emb_mpnet.npy"}[key]
        E=np.load(d+emb)
        U=json.load(open(d+r"\oncology\oncology_units.json",encoding="utf-8"))
        ss=lambda u:(u.get('l1',[]) or [])
        axes=['pathology','part','domain']; axget=lambda u,a:(u.get(a,[]) or [])
    E=E.astype(np.float64); E/=(np.linalg.norm(E,axis=1,keepdims=True)+1e-12)
    def labels(u):
        L=set()
        for p in ss(u): L.add('ss:'+p)
        for a in axes:
            for v in axget(u,a): L.add(f'{a}:{v}')
        return L
    return E, U, [labels(u) for u in U]

# ---- 공통: 발견 후보 + ρ ----
def candidates(E, seed, k_pool=200, z=1.5):
    km=KMeans(n_clusters=k_pool,random_state=seed,n_init=3).fit(E)
    pr=km.cluster_centers_; pr/=(np.linalg.norm(pr,axis=1,keepdims=True)+1e-12)
    s=E@pr.T; Z=(s-s.mean(1,keepdims=True))/(s.std(1,keepdims=True)+1e-12)
    B=(Z>z).astype(np.int8)
    return B

def sat_k(B, idx, target=0.95):
    n=len(idx); Bi=B[idx]
    active=[c for c in range(Bi.shape[1]) if 0<int(Bi[:,c].sum())<n]
    if not active: return None
    order=sorted(active,key=lambda c:-min(int(Bi[:,c].sum()),n-int(Bi[:,c].sum())))
    keys=[() for _ in range(n)]; goal=int(n*target)
    for k,c in enumerate(order,1):
        keys=[keys[i]+(int(Bi[i,c]),) for i in range(n)]
        if len(set(keys))>=goal: return k
    return len(order)

def fit_growth(B, NT, rng, Ns):
    data=[]
    for Nn in Ns:
        if Nn>NT: continue
        ks=[sat_k(B,rng.choice(NT,size=Nn,replace=False)) for _ in range(1)]
        ks=[k for k in ks if k]
        if ks: data.append((Nn,float(np.mean(ks))))
    if len(data)<3: return None
    Narr=np.array([d[0] for d in data],float); karr=np.array([d[1] for d in data],float)
    fits={}
    for name,f in [('log',np.log(Narr)),('linear',Narr),('sqrt',np.sqrt(Narr))]:
        A=np.vstack([f,np.ones_like(f)]).T
        coef,_,_,_=np.linalg.lstsq(A,karr,rcond=None)
        pred=A@coef; r2=1-((karr-pred)**2).sum()/((karr-karr.mean())**2).sum()
        fits[name]=float(r2)
    cs=[(n, k, k/np.log2(n)) for n,k in data]
    return dict(data=data, fits=fits, c=[c for _,_,c in cs])

DATASETS=['508-OpenAI','diverse-OpenAI','onco-OpenAI','onco-E5','onco-MPNet']
result={'seeds':SEEDS,'meta':{'k_pool':200,'z':1.5,'target':0.95}}
t0=time.time()

# ===== P3 + P1: 데이터셋별 성장률·c·천장 (5시드) =====
print("[P3/P1] 데이터셋별 5시드 측정...")
result['P3_sublinear']={}
for key in DATASETS:
    E,U,LAB=load(key); NT=E.shape[0]
    Ns=[80,160,240,320,400,480] if NT<600 else [100,200,400,800,1600,3200]
    per_seed_c=[]; per_seed_fits=[]; per_seed_data=[]
    for sd in SEEDS:
        B=candidates(E,sd)
        rng=np.random.default_rng(sd)
        g=fit_growth(B,NT,rng,Ns)
        if g:
            per_seed_c.append(g['c']); per_seed_fits.append(g['fits'])
            per_seed_data.append(g['data'])
        print(f"   {key} seed={sd} done [{time.time()-t0:.0f}s]")
    # c: N별 평균±std, 전체 평균±std
    carr=np.array(per_seed_c)  # seeds x Ngrid
    c_by_N=carr.mean(0); c_by_N_std=carr.std(0)
    logR2=np.array([f['log'] for f in per_seed_fits])
    sqR2=np.array([f['sqrt'] for f in per_seed_fits])
    linR2=np.array([f['linear'] for f in per_seed_fits])
    result['P3_sublinear'][key]=dict(
        N_total=NT, N_grid=Ns,
        c_mean=float(carr.mean()), c_std=float(carr.std()),
        c_by_N=[float(x) for x in c_by_N], c_by_N_std=[float(x) for x in c_by_N_std],
        logR2_mean=float(logR2.mean()), sqrtR2_mean=float(sqR2.mean()),
        linR2_mean=float(linR2.mean()),
        k_by_N=[[float(d[1]) for d in ds] for ds in per_seed_data])
    print(f"  {key}: c={carr.mean():.2f}±{carr.std():.2f}, "
          f"logR2={logR2.mean():.3f} linR2={linR2.mean():.3f}")

# ===== N: z×k_pool robustness (onco-OpenAI, 5시드) =====
print("\n[N] z×k_pool robustness (onco)...")
E,U,LAB=load('onco-OpenAI'); NT=E.shape[0]
Ns=[100,200,400,800,1600,3200]
result['N_robustness']={}
for (z,kp) in [(1.0,200),(1.5,200),(2.0,200),(2.5,200),(1.5,100),(1.5,400)]:
    best_list=[]; lin_list=[]; sub_list=[]
    for sd in SEEDS:
        B=candidates(E,sd,k_pool=kp,z=z)
        rng=np.random.default_rng(sd)
        g=fit_growth(B,NT,rng,Ns)
        if g:
            best=max(g['fits'],key=g['fits'].get); best_list.append(best)
            lin_list.append(g['fits']['linear'])
            sub_list.append(max(g['fits']['log'],g['fits']['sqrt']))
    result['N_robustness'][f'z{z}_k{kp}']=dict(
        linR2_mean=float(np.mean(lin_list)), subR2_mean=float(np.mean(sub_list)),
        sublinear_all=bool(all(b!='linear' for b in best_list)))
    print(f"   z={z} k={kp}: subR2={np.mean(sub_list):.3f} linR2={np.mean(lin_list):.3f} [{time.time()-t0:.0f}s]")

# ===== P4: 자연부류 (508, 5시드) =====
print("\n[P4] 자연부류 (508)...")
E,U,LAB=load('508-OpenAI'); N=E.shape[0]
def lab_coh(idx):
    if len(idx)<2: return None
    js=[]
    for a in range(len(idx)):
        for b in range(a+1,len(idx)):
            A,Bb=LAB[idx[a]],LAB[idx[b]]; u=len(A|Bb)
            js.append(len(A&Bb)/u if u else 0)
    return float(np.mean(js))
def emb_coh(idx):
    if len(idx)<2: return None
    V=E[idx]; G=V@V.T; iu=np.triu_indices(len(idx),1); return float(G[iu].mean())
lab_b=[]; lab_r=[]; emb_b=[]; emb_r=[]
for sd in SEEDS:
    B=candidates(E,sd); rng=np.random.default_rng(sd)
    active=[c for c in range(200) if 0<int(B[:,c].sum())<N]
    order=sorted(active,key=lambda c:-min(int(B[:,c].sum()),N-int(B[:,c].sum())))
    recs=[]
    def split(idx,d,fp):
        if fp>=len(order) or len(idx)<3: return
        c=order[fp]; on=[i for i in idx if B[i,c]>0]; off=[i for i in idx if B[i,c]==0]
        if not on or not off: split(idx,d,fp+1); return
        for ch in (on,off):
            if len(ch)>=3:
                lc=lab_coh(ch); ec=emb_coh(ch)
                rl=np.mean([lab_coh(rng.choice(N,len(ch),replace=False).tolist()) for _ in range(10)])
                re=np.mean([emb_coh(rng.choice(N,len(ch),replace=False).tolist()) for _ in range(10)])
                recs.append((lc,ec,rl,re))
        split(on,d+1,fp+1); split(off,d+1,fp+1)
    import sys; sys.setrecursionlimit(10000)
    split(list(range(N)),0,0)
    r=np.array(recs)
    lab_b.append(r[:,0].mean()); lab_r.append(r[:,2].mean())
    emb_b.append(r[:,1].mean()); emb_r.append(r[:,3].mean())
    print(f"   seed={sd} done [{time.time()-t0:.0f}s]")
result['P4_naturalclass']=dict(
    lab_branch_mean=float(np.mean(lab_b)), lab_branch_std=float(np.std(lab_b)),
    lab_rand_mean=float(np.mean(lab_r)),
    emb_branch_mean=float(np.mean(emb_b)), emb_branch_std=float(np.std(emb_b)),
    emb_rand_mean=float(np.mean(emb_r)))
print(f"  라벨: 분기 {np.mean(lab_b):.3f}±{np.std(lab_b):.3f} vs 무작위 {np.mean(lab_r):.3f}")

json.dump(result, open(OUT,'w'), indent=2)
print(f"\n[완료] {OUT}  ({time.time()-t0:.0f}s)")
print("이 파일 하나를 새 채팅에 올리면 §5 수치 전부 정합.")
