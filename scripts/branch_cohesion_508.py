"""
의미 분기 — 1단계 핵심 단위 (508)
==================================
공유자질 집합 S를 추가자질 f로 둘로 가름(S_on/S_off).
유효한 분할 = 두 하위집합이 S전체보다 임베딩상 응집(within 평균코사인 상승). 자질과 독립.
비교: f가 발견자질(군집) vs 기저자질(supersense) 일 때 응집향상.
사용자 가설: 기저자질 분할이 더 유효(응집향상 큼) 또는 발견 포화 너머서 유효 공급.
"""
import numpy as np, json
from sklearn.cluster import KMeans

EMB=np.load('/mnt/user-data/uploads/news_legal_embeddings.npy').astype(np.float64)
UNITS=json.load(open('/mnt/user-data/uploads/news_legal_units.json'))
N=EMB.shape[0]

# 기저자질: supersense 이진
def ss_of(u):
    l=u.get('l1',''); return l.split('|') if l else []
SS=sorted({x for u in UNITS for x in ss_of(u)}); si={s:j for j,s in enumerate(SS)}
WN=np.zeros((N,len(SS)),dtype=np.int8)
for i,u in enumerate(UNITS):
    for p in ss_of(u): WN[i,si[p]]=1

# 발견자질
km=KMeans(n_clusters=200,random_state=42,n_init=3).fit(EMB)
pr=km.cluster_centers_; pr/=(np.linalg.norm(pr,axis=1,keepdims=True)+1e-12)
s=EMB@pr.T; Z=(s-s.mean(1,keepdims=True))/(s.std(1,keepdims=True)+1e-12)
Bd=(Z>1.5).astype(np.int8)
DISC=[Bd[:,c] for c in range(200) if Bd[:,c].sum()>0]

def cohesion(idx):
    """집합 내 평균 쌍 코사인(임베딩 정규화됨 → 내적). 응집도."""
    if len(idx)<2: return None
    V=EMB[idx]
    G=V@V.T
    iu=np.triu_indices(len(idx),1)
    return float(G[iu].mean())

def split_gain(S_idx, feat):
    """S를 feat로 on/off 분할. 향상 = 가중평균(within) - 원래 within."""
    on=[i for i in S_idx if feat[i]>0]; off=[i for i in S_idx if feat[i]==0]
    if len(on)<2 or len(off)<2: return None
    c0=cohesion(S_idx)
    con=cohesion(on); coff=cohesion(off)
    w=(len(on)*con+len(off)*coff)/(len(on)+len(off))
    return w-c0, len(on), len(off)

# 루트(공유자질 집합) 후보: 각 발견자질이 켜는 토큰집합을 S로 사용(현실적 공유집합)
# 그 S를 '다른' 발견자질 vs 기저자질로 가름.
print("508 — 공유자질 집합을 발견/기저자질로 가를 때 응집향상")
print("="*60)

roots=[set(np.where(DISC[c]>0)[0]) for c in range(min(40,len(DISC)))]
roots=[r for r in roots if len(r)>=20]   # 충분히 큰 집합만
print(f"루트(공유자질 집합) {len(roots)}개 (크기≥20)")

disc_gains=[]; wn_gains=[]
for S in roots:
    Sl=sorted(S)
    # 발견자질로 가름 (S를 만든 것 외 다른 발견자질들)
    for c in range(len(DISC)):
        g=split_gain(Sl, DISC[c])
        if g and g[1]>=5 and g[2]>=5: disc_gains.append(g[0])
    # 기저자질로 가름
    for j in range(len(SS)):
        g=split_gain(Sl, WN[:,j])
        if g and g[1]>=5 and g[2]>=5: wn_gains.append(g[0])

disc_gains=np.array(disc_gains); wn_gains=np.array(wn_gains)
print(f"\n발견자질 분할: n={len(disc_gains)}, 평균 응집향상 {disc_gains.mean():+.4f}, "
      f"양수비율 {(disc_gains>0).mean():.2f}")
print(f"기저자질 분할: n={len(wn_gains)}, 평균 응집향상 {wn_gains.mean():+.4f}, "
      f"양수비율 {(wn_gains>0).mean():.2f}")
print(f"\n향상 분포(기저): 중앙 {np.median(wn_gains):+.4f}, 상위10% {np.percentile(wn_gains,90):+.4f}")
print(f"향상 분포(발견): 중앙 {np.median(disc_gains):+.4f}, 상위10% {np.percentile(disc_gains,90):+.4f}")
from scipy.stats import mannwhitneyu
u,p=mannwhitneyu(wn_gains,disc_gains,alternative='greater')
print(f"\nMann-Whitney(기저>발견 응집향상) p={p:.3e}")
print("→ 기저가 크면: supersense 분할이 임베딩상 더 유효한 의미무리 생성 = 사용자가설")
print("→ 발견이 크면: 군집 분할이 더 유효 (기저는 이미 군집에 포함)")

json.dump(dict(disc_mean=float(disc_gains.mean()),wn_mean=float(wn_gains.mean()),
               disc_pos=float((disc_gains>0).mean()),wn_pos=float((wn_gains>0).mean()),
               mwu_p=float(p)),
          open('/home/claude/branch_cohesion_508.json','w'))
print("\n[saved]")
