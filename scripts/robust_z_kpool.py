"""
하이퍼파라미터 robustness — z 임계 × k_pool (oncology)
=====================================================
예측 P3(sublinear 수렴)·P1(천장 존재)이 z·k_pool의 우연이 아님을 보증.
반증가능성의 전제: 자의적 하이퍼파라미터에 의존하면 예측이 아니라 아티팩트.
측정: 각 (z, k_pool) 조합에서 N부분표본 포화 k 성장률 → log/linear 피팅.
모든 조합서 sublinear(선형 기각) 유지면 예측이 하이퍼파라미터 불변.
"""
import numpy as np, json
from sklearn.cluster import KMeans

EMB=np.load('/mnt/user-data/uploads/oncology_embeddings.npy').astype(np.float64)
NT=EMB.shape[0]
rng=np.random.default_rng(0)

def candidates(k_pool, z):
    km=KMeans(n_clusters=k_pool,random_state=42,n_init=3).fit(EMB)
    pr=km.cluster_centers_; pr/=(np.linalg.norm(pr,axis=1,keepdims=True)+1e-12)
    s=EMB@pr.T; Z=(s-s.mean(1,keepdims=True))/(s.std(1,keepdims=True)+1e-12)
    return (Z>z).astype(np.int8)

def sat_k(Bf, idx, target=0.95):
    B=Bf[idx]; n=len(idx)
    active=[c for c in range(B.shape[1]) if 0<int(B[:,c].sum())<n]
    if not active: return None
    order=sorted(active,key=lambda c:-min(int(B[:,c].sum()),n-int(B[:,c].sum())))
    keys=[() for _ in range(n)]; goal=int(n*target)
    for k,c in enumerate(order,1):
        bits=B[:,c]; keys=[keys[i]+(int(bits[i]),) for i in range(n)]
        if len(set(keys))>=goal: return k
    return len(order)

def growth_fit(Bf):
    Ns=[100,200,400,800,1600,3200]
    data=[]
    for Nn in Ns:
        ks=[sat_k(Bf,rng.choice(NT,size=Nn,replace=False)) for _ in range(3)]
        ks=[k for k in ks if k]
        if ks: data.append((Nn,np.mean(ks)))
    Narr=np.array([d[0] for d in data],float); karr=np.array([d[1] for d in data],float)
    fits={}
    for name,f in [('log',np.log(Narr)),('linear',Narr),('sqrt',np.sqrt(Narr))]:
        A=np.vstack([f,np.ones_like(f)]).T
        coef,_,_,_=np.linalg.lstsq(A,karr,rcond=None)
        pred=A@coef; fits[name]=1-((karr-pred)**2).sum()/((karr-karr.mean())**2).sum()
    return data, fits

print("oncology — z × k_pool robustness")
print("="*64)
print(f"{'z':>4} {'k_pool':>7} | {'k@100':>6} {'k@3200':>7} | {'logR2':>6} {'sqrtR2':>7} {'linR2':>6} | 최적")
print("-"*64)
results={}
# 1) z 스윕 (k_pool=200 고정)
print("[z 스윕, k_pool=200]")
for z in [1.0,1.5,2.0,2.5]:
    Bf=candidates(200,z)
    data,fits=growth_fit(Bf)
    k100=data[0][1]; k3200=data[-1][1]
    best=max(fits,key=fits.get)
    print(f"{z:>4} {200:>7} | {k100:>6.1f} {k3200:>7.1f} | "
          f"{fits['log']:.3f} {fits['sqrt']:.3f}  {fits['linear']:.3f} | {best}")
    results[f'z{z}_k200']=dict(data=data,fits=fits,best=best)

# 2) k_pool 스윕 (z=1.5 고정)
print("[k_pool 스윕, z=1.5]")
for kp in [100,200,400]:
    Bf=candidates(kp,1.5)
    data,fits=growth_fit(Bf)
    k100=data[0][1]; k3200=data[-1][1]
    best=max(fits,key=fits.get)
    print(f"{1.5:>4} {kp:>7} | {k100:>6.1f} {k3200:>7.1f} | "
          f"{fits['log']:.3f} {fits['sqrt']:.3f}  {fits['linear']:.3f} | {best}")
    results[f'z1.5_k{kp}']=dict(data=data,fits=fits,best=best)

print("\n[판정]")
all_sublinear=all(r['best']!='linear' and r['fits']['linear']<max(r['fits']['log'],r['fits']['sqrt'])-0.03
                  for r in results.values())
lin_max=max(r['fits']['linear'] for r in results.values())
sub_min=min(max(r['fits']['log'],r['fits']['sqrt']) for r in results.values())
print(f"  모든 조합 sublinear(선형 기각)? {all_sublinear}")
print(f"  선형 R² 최대 {lin_max:.3f} vs sublinear R² 최소 {sub_min:.3f}")
print(f"  → 모든 z·k_pool서 sublinear 유지면 P3 예측 하이퍼파라미터 불변(반증가능성 전제 충족)")

json.dump({k:{'best':v['best'],'fits':{f:round(r,4) for f,r in v['fits'].items()}}
           for k,v in results.items()},
          open('/home/claude/robust_z_kpool.json','w'),indent=2)
print("[saved] robust_z_kpool.json")
