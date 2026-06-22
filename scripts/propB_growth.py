"""
명제 B 검증 — 포화 k가 토큰 수 N에 따라 어떻게 자라나 (성장률)
=============================================================
명제 A(자명): 실재 의미분화 ≤ N. 검증 불필요.
명제 B(비자명·핵심): N개 토큰을 다 구별하는 데 필요한 자질 k가 작나?
  log N이면 FSPB 실용적(8000토큰→~100자질), 선형이면 무의미.
설계: oncology 전체(4629) 임베딩에서 부분표본 N 키우며 포화 k 측정.
  N=100,200,400,800,1600,3200. 각 N에서 distinct 비트열이 N의 95% 도달하는 k.
  k vs N을 log/linear/sqrt 피팅 → 어느 것인지.
같은 임베딩·자질풀 → N만의 효과 분리. (루트 supersense 안 씀 — 전체 토큰 대상,
  '의미 변별'의 일반 한계를 봄)
"""
import numpy as np, json
from sklearn.cluster import KMeans

EMB=np.load('/mnt/user-data/uploads/oncology_embeddings.npy').astype(np.float64)
NTOT=EMB.shape[0]
print(f"oncology 전체 {NTOT}토큰")

# 전체에서 발견자질 후보 (한 번만)
km=KMeans(n_clusters=300,random_state=42,n_init=3).fit(EMB)
pr=km.cluster_centers_; pr/=(np.linalg.norm(pr,axis=1,keepdims=True)+1e-12)
sim=EMB@pr.T; Z=(sim-sim.mean(1,keepdims=True))/(sim.std(1,keepdims=True)+1e-12)
Bfull=(Z>1.5).astype(np.int8)   # NTOT x 300

def saturation_k(idx, target=0.95):
    """idx 토큰들을 자질 순번(균형순)대로 더하며 distinct 비트열이
    len(idx)*target 도달하는 k. 자질 순서는 이 부분집합 기준 균형순."""
    B=Bfull[idx]   # n x 300
    n=len(idx)
    active=[c for c in range(300) if 0<int(B[:,c].sum())<n]
    order=sorted(active,key=lambda c:-min(int(B[:,c].sum()),n-int(B[:,c].sum())))
    keys=[() for _ in range(n)]
    goal=int(n*target)
    for k,c in enumerate(order,1):
        bits=B[:,c]
        keys=[keys[i]+(int(bits[i]),) for i in range(n)]
        if len(set(keys))>=goal:
            return k
    return len(order)  # 다 써도 목표 미달

rng=np.random.default_rng(0)
Ns=[100,200,400,800,1600,3200]
print("\nN별 포화 k (distinct 비트열이 N의 95% 도달, 3표본 평균):")
data=[]
for Nn in Ns:
    if Nn>NTOT: continue
    ks=[]
    for rep in range(3):
        idx=rng.choice(NTOT,size=Nn,replace=False)
        ks.append(saturation_k(idx))
    kmean=np.mean(ks)
    data.append((Nn,kmean))
    print(f"  N={Nn:5d}: k={kmean:.1f}  (표본 {ks})")

# 피팅: k vs N — log, linear, sqrt 중 무엇
Narr=np.array([d[0] for d in data],float)
karr=np.array([d[1] for d in data],float)
print("\n[성장률 피팅] k = a·f(N) + b")
fits={}
for name,f in [('log',np.log(Narr)),('linear',Narr),('sqrt',np.sqrt(Narr))]:
    A=np.vstack([f,np.ones_like(f)]).T
    coef,res,_,_=np.linalg.lstsq(A,karr,rcond=None)
    pred=A@coef
    ss_res=((karr-pred)**2).sum(); ss_tot=((karr-karr.mean())**2).sum()
    r2=1-ss_res/ss_tot
    fits[name]=r2
    print(f"  {name:7s}: R²={r2:.4f}  (k≈{coef[0]:.1f}·{name}(N)+{coef[1]:.1f})")
best=max(fits,key=fits.get)
print(f"\n  → 최적 적합: {best} (R²={fits[best]:.4f})")
if best=='log':
    print("  ★명제B 입증: 포화 k가 log(N) — 토큰 폭증해도 자질 천장 로그로만 자람.")
    print("    8000토큰도 ~100자질로 구별. FSPB 실용성·128이내의 근본 이유.")
elif best=='sqrt':
    print("  포화 k가 √N — 로그보단 빠르나 선형보단 느림. 부분적 실용성.")
else:
    print("  포화 k가 선형 — 이진화 압축 이득 약함. 명제B 약화.")

json.dump(dict(data=[(int(n),float(k)) for n,k in data],fits=fits,best=best),
          open('/home/claude/propB_growth.json','w'))
print("\n[saved] propB_growth.json")
