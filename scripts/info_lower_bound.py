"""
정보이론 하한 대비 효율계수 — 고정폭의 형식화
==============================================
N개 토큰 구별의 정보이론 하한 = log2(N) 비트(절대 하한).
실측 고정폭 k 대비: 효율계수 c = k / log2(N).
c가 N에 걸쳐 상수면 → k ≈ c·log2(N), 정보이론 하한의 상수배 = 점근 거의최적.
"128 이내" 임의상수 대신 유도된 정량법칙.
도메인·모델·z 걸쳐 c 안정성 측정.
"""
import numpy as np, json
from sklearn.cluster import KMeans

def sat_k(Bf, idx, NT, target=0.95):
    B=Bf[idx]; n=len(idx)
    active=[c for c in range(B.shape[1]) if 0<int(B[:,c].sum())<n]
    if not active: return None
    order=sorted(active,key=lambda c:-min(int(B[:,c].sum()),n-int(B[:,c].sum())))
    keys=[() for _ in range(n)]; goal=int(n*target)
    for k,c in enumerate(order,1):
        bits=B[:,c]; keys=[keys[i]+(int(bits[i]),) for i in range(n)]
        if len(set(keys))>=goal: return k
    return len(order)

def candidates(EMB,k_pool=200,z=1.5):
    km=KMeans(n_clusters=k_pool,random_state=42,n_init=3).fit(EMB)
    pr=km.cluster_centers_; pr/=(np.linalg.norm(pr,axis=1,keepdims=True)+1e-12)
    s=EMB@pr.T; Z=(s-s.mean(1,keepdims=True))/(s.std(1,keepdims=True)+1e-12)
    return (Z>z).astype(np.int8)

def coeff_curve(EMB, label, z=1.5):
    NT=EMB.shape[0]
    EMB=EMB/(np.linalg.norm(EMB,axis=1,keepdims=True)+1e-12)
    Bf=candidates(EMB,200,z)
    rng=np.random.default_rng(0)
    Ns=[n for n in [100,200,400,800,1600,3200] if n<=NT]
    rows=[]
    for Nn in Ns:
        ks=[sat_k(Bf,rng.choice(NT,size=Nn,replace=False),NT) for _ in range(3)]
        ks=[k for k in ks if k]
        if not ks: continue
        k=np.mean(ks); lb=np.log2(Nn); c=k/lb
        rows.append((Nn,k,lb,c))
    return rows

datasets={
 '508-OpenAI':'/mnt/user-data/uploads/news_legal_embeddings.npy',
 'diverse-OpenAI':'/mnt/user-data/uploads/news_diverse_embeddings.npy',
 'onco-OpenAI':'/mnt/user-data/uploads/oncology_embeddings.npy',
 'onco-E5':'/mnt/user-data/uploads/oncology_emb_e5.npy',
 'onco-MPNet':'/mnt/user-data/uploads/oncology_emb_mpnet.npy',
}

print("효율계수 c = k / log2(N)  — 상수면 k≈c·log2(N) (정보이론 하한의 c배)")
print("="*70)
allres={}
for label,path in datasets.items():
    EMB=np.load(path).astype(np.float64)
    rows=coeff_curve(EMB,label)
    cs=[r[3] for r in rows]
    print(f"\n{label} (NT={EMB.shape[0]}):")
    print(f"  {'N':>5} {'k':>6} {'log2N':>6} {'c=k/log2N':>10}")
    for Nn,k,lb,c in rows:
        print(f"  {Nn:>5} {k:>6.1f} {lb:>6.2f} {c:>10.2f}")
    print(f"  c 범위 [{min(cs):.2f}, {max(cs):.2f}], 평균 {np.mean(cs):.2f}, "
          f"변동계수(CV) {np.std(cs)/np.mean(cs):.2f}")
    allres[label]=dict(rows=rows,c_mean=float(np.mean(cs)),
                       c_cv=float(np.std(cs)/np.mean(cs)))

print("\n"+"="*70)
print("판정: c가 상수(낮은 CV)면 k≈c·log2N 정량법칙. c가 N따라 발산하면 반증.")
print("="*70)
for label,r in allres.items():
    trend = "상수적" if r['c_cv']<0.25 else "변동"
    # c가 N따라 증가/감소 추세?
    rows=r['rows']; cs=[x[3] for x in rows]
    slope = (cs[-1]-cs[0])/(np.log2(rows[-1][0])-np.log2(rows[0][0]))
    print(f"  {label:16s}: c평균 {r['c_mean']:.2f}, CV {r['c_cv']:.2f} ({trend}), "
          f"c기울기(vs log2N) {slope:+.2f}")
print("\n  CV<0.25면 c 거의상수 = 정보이론 하한의 상수배 점근최적.")
print("  c기울기≈0이면 순수 log, 양수면 log보다 약간 빠름(하지만 sublinear).")

json.dump(allres,open('/home/claude/info_lower_bound.json','w'),default=str,indent=2)
print("\n[saved] info_lower_bound.json")
