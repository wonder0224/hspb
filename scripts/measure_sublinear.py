"""
measure_sublinear.py  —  [Claude 환경 또는 당신 PC]
====================================================
임의 임베딩(.npy)의 자질복잡도 성장률 측정. K2/K3와 동일 방법.
모델 독립성: text-embedding-3-large 결과와 BERT/E5 결과를 같은 측정으로 비교.

사용:
  python measure_sublinear.py <embedding.npy> [model_name]
  예: python measure_sublinear.py oncology_emb_e5.npy E5
      python measure_sublinear.py oncology_embeddings.npy OpenAI-3-large

측정: N부분표본(100~3200) 키우며 포화 k(distinct비트열 95%) → log/sqrt/linear 피팅.
sublinear(log 최적, 선형 기각)면 그 모델도 명제B 만족.
"""
import sys, json, numpy as np
from sklearn.cluster import KMeans

EMB_PATH = sys.argv[1] if len(sys.argv)>1 else 'oncology_embeddings.npy'
MODEL    = sys.argv[2] if len(sys.argv)>2 else 'unknown'

EMB = np.load(EMB_PATH).astype(np.float64)
# L2 정규화 보장 (모델마다 다를 수 있음)
EMB /= (np.linalg.norm(EMB,axis=1,keepdims=True)+1e-12)
NT = EMB.shape[0]
print(f"[{MODEL}] {EMB_PATH}: shape {EMB.shape}")

# 발견자질 후보 (동일 절차: KMeans 300 → z=1.5 이진화)
km = KMeans(n_clusters=300, random_state=42, n_init=3).fit(EMB)
pr = km.cluster_centers_; pr /= (np.linalg.norm(pr,axis=1,keepdims=True)+1e-12)
s = EMB@pr.T; Z = (s-s.mean(1,keepdims=True))/(s.std(1,keepdims=True)+1e-12)
Bf = (Z>1.5).astype(np.int8)

def sat_k(idx, target=0.95):
    B=Bf[idx]; n=len(idx)
    active=[c for c in range(B.shape[1]) if 0<int(B[:,c].sum())<n]
    order=sorted(active,key=lambda c:-min(int(B[:,c].sum()),n-int(B[:,c].sum())))
    keys=[() for _ in range(n)]; goal=int(n*target)
    for k,c in enumerate(order,1):
        bits=B[:,c]; keys=[keys[i]+(int(bits[i]),) for i in range(n)]
        if len(set(keys))>=goal: return k
    return len(order)

rng=np.random.default_rng(0)
Ns=[n for n in [100,200,400,800,1600,3200] if n<=NT]
data=[]
print("\nN별 포화 k (3표본 평균):")
for Nn in Ns:
    ks=[sat_k(rng.choice(NT,size=Nn,replace=False)) for _ in range(3)]
    data.append((Nn,float(np.mean(ks))))
    print(f"  N={Nn:5d}: k={np.mean(ks):.1f}  std={np.std(ks):.1f}")

Narr=np.array([d[0] for d in data],float); karr=np.array([d[1] for d in data],float)
print("\n피팅 R²:")
fits={}
for name,f in [('log',np.log(Narr)),('linear',Narr),('sqrt',np.sqrt(Narr))]:
    A=np.vstack([f,np.ones_like(f)]).T
    coef,_,_,_=np.linalg.lstsq(A,karr,rcond=None)
    pred=A@coef; r2=1-((karr-pred)**2).sum()/((karr-karr.mean())**2).sum()
    fits[name]=float(r2)
    print(f"  {name:7s}: R²={r2:.4f}")
best=max(fits,key=fits.get)
print(f"\n[{MODEL}] 최적={best}, 선형R²={fits['linear']:.3f}")
if best in ('log','sqrt') and fits['linear']<fits[best]-0.05:
    print(f"  ★ {MODEL}도 sublinear({best}). 명제B 모델독립 지지.")
else:
    print(f"  {MODEL}은 sublinear 약함/선형. 모델독립 반례 가능 — 재검토.")

json.dump(dict(model=MODEL,path=EMB_PATH,N_total=NT,data=data,fits=fits,best=best),
          open(f'sublinear_{MODEL}.json','w'))
print(f"[saved] sublinear_{MODEL}.json")
