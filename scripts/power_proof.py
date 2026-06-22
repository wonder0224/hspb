"""
검정력 입증 — 작은 N(좁은 범위)이 로그 피팅에 불리함을 같은 데이터로 증명
==========================================================================
주장: 508의 낮은 R²(0.917)는 데이터 탓 아니라 좁은 N범위(검정력 부족).
증명: 같은 oncology를
  (a) 넓은 범위 100~3200 (log축 폭 3.47)
  (b) 좁은 범위 100~500  (508과 유사, log축 폭 1.6)
  로 피팅 → (b)의 log R²가 (a)보다 떨어지면, 범위폭이 원인.
+ 분산도 비교: 작은 N의 k 표본분산이 큰가.
+ 다표본(10회)으로 R² 자체의 분산도.
"""
import numpy as np, json
from sklearn.cluster import KMeans

EMB=np.load('/mnt/user-data/uploads/oncology_embeddings.npy').astype(np.float64)
NT=EMB.shape[0]
km=KMeans(n_clusters=300,random_state=42,n_init=3).fit(EMB)
pr=km.cluster_centers_; pr/=(np.linalg.norm(pr,axis=1,keepdims=True)+1e-12)
s=EMB@pr.T; Z=(s-s.mean(1,keepdims=True))/(s.std(1,keepdims=True)+1e-12)
Bf=(Z>1.5).astype(np.int8)

def sat_k(idx,target=0.95):
    B=Bf[idx]; n=len(idx)
    active=[c for c in range(B.shape[1]) if 0<int(B[:,c].sum())<n]
    order=sorted(active,key=lambda c:-min(int(B[:,c].sum()),n-int(B[:,c].sum())))
    keys=[() for _ in range(n)]; goal=int(n*target)
    for k,c in enumerate(order,1):
        bits=B[:,c]; keys=[keys[i]+(int(bits[i]),) for i in range(n)]
        if len(set(keys))>=goal: return k
    return len(order)

rng=np.random.default_rng(0)
def fit_r2(Ns,reps=5):
    data=[]; var_by_N={}
    for Nn in Ns:
        ks=[sat_k(rng.choice(NT,size=Nn,replace=False)) for _ in range(reps)]
        data.append((Nn,np.mean(ks))); var_by_N[Nn]=np.std(ks)
    Narr=np.array([d[0] for d in data],float); karr=np.array([d[1] for d in data],float)
    r2={}
    for name,f in [('log',np.log(Narr)),('linear',Narr),('sqrt',np.sqrt(Narr))]:
        A=np.vstack([f,np.ones_like(f)]).T
        coef,_,_,_=np.linalg.lstsq(A,karr,rcond=None)
        pred=A@coef; r2[name]=1-((karr-pred)**2).sum()/((karr-karr.mean())**2).sum()
    return r2,var_by_N,data

print("같은 oncology, 범위만 다르게 — log 피팅 R² 비교")
print("="*56)
wide=[100,200,400,800,1600,3200]
narrow=[100,200,300,400,500]   # 508 크기와 유사한 좁은 범위

r2w,varw,dw=fit_r2(wide)
r2n,varn,dn=fit_r2(narrow)

print(f"\n(a) 넓은 범위 {wide[0]}~{wide[-1]} (log폭 {np.log(wide[-1]/wide[0]):.2f}):")
print(f"    log R²={r2w['log']:.4f}  sqrt={r2w['sqrt']:.4f}  linear={r2w['linear']:.4f}")
print(f"(b) 좁은 범위 {narrow[0]}~{narrow[-1]} (log폭 {np.log(narrow[-1]/narrow[0]):.2f}):")
print(f"    log R²={r2n['log']:.4f}  sqrt={r2n['sqrt']:.4f}  linear={r2n['linear']:.4f}")
print(f"\n  → 좁은 범위 log R²({r2n['log']:.3f}) < 넓은({r2w['log']:.3f}) 면:")
print(f"     낮은 R²는 데이터 아니라 범위폭(검정력) 탓 = 508 불리함의 원인 입증")
print(f"  → 좁은 범위선 log/sqrt/linear R² 서로 가까움(구분 안 됨):")
print(f"     좁={sorted([round(v,3) for v in r2n.values()])} vs 넓={sorted([round(v,3) for v in r2w.values()])}")

print(f"\n표본 분산 (k의 std, 작은 N일수록 큰가):")
for Nn in sorted(set(wide)):
    if Nn in varw: print(f"   N={Nn:5d}: std(k)={varw[Nn]:.1f}")

# R² 자체의 분산 (좁은 범위에서 R²가 run마다 흔들리나)
print(f"\nR² 안정성 (10회 재표본, 좁은 범위):")
r2n_log=[]
for _ in range(10):
    r2,_,_=fit_r2(narrow,reps=3); r2n_log.append(r2['log'])
print(f"   좁은범위 log R²: 평균 {np.mean(r2n_log):.3f} std {np.std(r2n_log):.3f} "
      f"범위 [{min(r2n_log):.3f},{max(r2n_log):.3f}]")
print(f"   → std 크면 좁은범위 R²가 불안정 = 검정력 부족 확증")

json.dump(dict(wide=r2w,narrow=r2n,var_wide=varw,
               r2_narrow_runs=r2n_log),
          open('/home/claude/power_proof.json','w'),default=str)
print("\n[saved] power_proof.json")
