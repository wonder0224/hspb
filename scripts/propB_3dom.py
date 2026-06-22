"""
명제 B 성장률 — 3도메인 나란히 (508 / oncology / diverse)
=========================================================
같은 측정(N 부분표본 키우며 포화 k)을 세 도메인에 적용, 곡선 겹치나.
508은 508개라 N 작게(80~480), onco/diverse는 크게.
sublinear(log~sqrt)가 세 도메인 일관이면 도메인 독립.
"""
import numpy as np, json

def setup(path):
    EMB=np.load(path).astype(np.float64)
    return EMB, EMB.shape[0]

from sklearn.cluster import KMeans
def candidates(EMB):
    km=KMeans(n_clusters=300,random_state=42,n_init=3).fit(EMB)
    pr=km.cluster_centers_; pr/=(np.linalg.norm(pr,axis=1,keepdims=True)+1e-12)
    s=EMB@pr.T; Z=(s-s.mean(1,keepdims=True))/(s.std(1,keepdims=True)+1e-12)
    return (Z>1.5).astype(np.int8)

def sat_k(Bfull, idx, target=0.95):
    B=Bfull[idx]; n=len(idx)
    active=[c for c in range(B.shape[1]) if 0<int(B[:,c].sum())<n]
    order=sorted(active,key=lambda c:-min(int(B[:,c].sum()),n-int(B[:,c].sum())))
    keys=[() for _ in range(n)]; goal=int(n*target)
    for k,c in enumerate(order,1):
        bits=B[:,c]; keys=[keys[i]+(int(bits[i]),) for i in range(n)]
        if len(set(keys))>=goal: return k
    return len(order)

doms={
 '508':'/mnt/user-data/uploads/news_legal_embeddings.npy',
 'onco':'/mnt/user-data/uploads/oncology_embeddings.npy',
 'diverse':'/mnt/user-data/uploads/news_diverse_embeddings.npy',
}
rng=np.random.default_rng(0)
allres={}
for dom,path in doms.items():
    EMB,NT=setup(path)
    Bf=candidates(EMB)
    # N 범위: 도메인 크기에 맞춰
    if NT<600: Ns=[80,160,240,320,400,480]
    else: Ns=[100,200,400,800,1600,3200]
    Ns=[n for n in Ns if n<=NT]
    data=[]
    for Nn in Ns:
        ks=[sat_k(Bf,rng.choice(NT,size=Nn,replace=False)) for _ in range(3)]
        data.append((Nn,float(np.mean(ks))))
    Narr=np.array([d[0] for d in data],float); karr=np.array([d[1] for d in data],float)
    fits={}
    for name,f in [('log',np.log(Narr)),('linear',Narr),('sqrt',np.sqrt(Narr))]:
        A=np.vstack([f,np.ones_like(f)]).T
        coef,_,_,_=np.linalg.lstsq(A,karr,rcond=None)
        pred=A@coef; r2=1-((karr-pred)**2).sum()/((karr-karr.mean())**2).sum()
        fits[name]=(round(r2,4),round(coef[0],2),round(coef[1],2))
    allres[dom]={'N_total':NT,'data':data,'fits':fits}
    print(f"=== {dom} (전체 {NT}) ===")
    for n,k in data: print(f"   N={n:5d}: k={k:.1f}")
    print(f"   피팅 R²: log={fits['log'][0]} linear={fits['linear'][0]} sqrt={fits['sqrt'][0]}")
    best=max(['log','linear','sqrt'],key=lambda x:fits[x][0])
    print(f"   최적: {best}\n")

# 세 도메인 곡선 정규화 비교 (k/√N 가 상수면 sqrt 일관)
print("="*56)
print("도메인 독립성: k 대 √N (sqrt면 k/√N≈상수)")
print("="*56)
for dom in doms:
    print(f"  {dom:8s}:", end=" ")
    for n,k in allres[dom]['data']:
        print(f"N{n}:{k/np.sqrt(n):.2f}", end="  ")
    print()
print("  → 각 도메인 내 k/√N가 비슷하면 sqrt, 도메인 간 값 비슷하면 도메인독립")

json.dump(allres,open('/home/claude/propB_3dom.json','w'),ensure_ascii=False,indent=2,default=str)
print("\n[saved] propB_3dom.json")
