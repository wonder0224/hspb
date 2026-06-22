"""
기저 투사의 해석 견고성 — (2)배열보존 + (1)라벨안정 (diverse)
=============================================================
명제(사용자): 기저자질(supersense)은 발견자질의 해석을 단단하게 만든다.
  변별은 발견이 독점(기저 잉여)이나, 해석은 기저가 고정점 제공 → 시드불변.
시험: 시드 흔들어 발견자질 여러번 뽑음.
  (1) 라벨안정: 발견자질→supersense 투사 라벨이 시드간 일치하나(기저 있을 때).
       대조: 발견자질끼리 직접 매칭(기저 없을 때) 안정성.
  (2) 배열보존: 토큰별 '해석 배열'이 시드간 보존되나.
       NoBasis 배열 = 발견자질 ID 집합(시드마다 ID 의미 달라 불안정).
       Basis 배열   = 발견자질의 supersense 라벨 집합(기저 투사로 안정 기대).
견고성 = 시드쌍 간 토큰 배열 Jaccard 평균. Basis > NoBasis면 입증.
"""
import numpy as np, json
from scipy.stats import rankdata
from sklearn.cluster import KMeans

EMB=np.load('/mnt/user-data/uploads/news_diverse_embeddings.npy').astype(np.float64)
UNITS=json.load(open('/mnt/user-data/uploads/news_diverse_units.json'))
N=EMB.shape[0]
tok_ss=[u.get('ontological',[]) or [] for u in UNITS]
SS=sorted({x for p in tok_ss for x in p}); ss_idx={s:j for j,s in enumerate(SS)}
WN=np.zeros((N,len(SS)),dtype=np.int8)
for i,parts in enumerate(tok_ss):
    for p in parts: WN[i,ss_idx[p]]=1
base=WN.sum(0)/N

rng=np.random.default_rng(0)
PAIRS=rng.integers(0,N,size=(15000,2)); PAIRS=PAIRS[PAIRS[:,0]!=PAIRS[:,1]]
PI,PJ=PAIRS[:,0],PAIRS[:,1]
COS=np.einsum('ij,ij->i',EMB[PI],EMB[PJ])
CR=rankdata(COS); CR=CR-CR.mean(); CRN=CR/(np.linalg.norm(CR)+1e-12)
def rho(M):
    if M.shape[1]==0: return 0.0
    a=M[PI]; b=M[PJ]; inter=(a&b).sum(1); union=(a|b).sum(1)
    with np.errstate(invalid='ignore',divide='ignore'):
        j=np.where(union>0,inter/union,0.0)
    if j.std()<1e-12: return 0.0
    jr=rankdata(j); jr=jr-jr.mean(); jn=jr/(np.linalg.norm(jr)+1e-12)
    return float(jn@CRN)

def discover(seed, max_feat=40):
    """발견자질(N차원 이진) 리스트 반환."""
    km=KMeans(n_clusters=200,random_state=seed,n_init=3).fit(EMB)
    pr=km.cluster_centers_; pr/=(np.linalg.norm(pr,axis=1,keepdims=True)+1e-12)
    s=EMB@pr.T; Z=(s-s.mean(1,keepdims=True))/(s.std(1,keepdims=True)+1e-12)
    Bd=(Z>1.5).astype(np.int8)
    cands=[Bd[:,c] for c in range(200) if Bd[:,c].sum()>0]
    M=np.array(cands).T
    sel=[]; cur=0.0; rem=list(range(M.shape[1]))
    while rem and len(sel)<max_feat:
        bc,br=None,cur
        for c in rem:
            r=rho(M[:,sel+[c]])
            if r>br: br,bc=r,c
        if bc is None: break
        sel.append(bc); rem.remove(bc)
        if br-cur<0.001 and len(sel)>=3: break
        cur=br
    return [cands[k] for k in sel]

def label_of(feat):
    """발견자질 → 최빈 supersense (보정), adj 임계 넘으면 라벨, 아니면 None."""
    on=feat.astype(bool)
    if on.sum()==0: return None
    mass=WN[on].mean(0); top=int(mass.argmax())
    if mass[top]-base[top] < 0.1: return None   # 정렬 약하면 라벨 안 붙임
    return SS[top]

SEEDS=[42,7,123,2024]
print(f"diverse N={N}. 시드 {len(SEEDS)}개로 발견 반복.")
feats_by_seed={}
labels_by_seed={}
for sd in SEEDS:
    fs=discover(sd)
    feats_by_seed[sd]=fs
    labels_by_seed[sd]=[label_of(f) for f in fs]
    nlab=sum(1 for l in labels_by_seed[sd] if l)
    print(f"  seed={sd}: 발견자질 {len(fs)}개, 라벨부착 {nlab}개")

# ---- 토큰별 배열 만들기 ----
def token_arrays_nobasis(feats):
    """토큰 → 켜진 발견자질 ID집합. ID는 '그 시드 내 순번'이라 시드간 의미 다름."""
    arrs=[set() for _ in range(N)]
    for fid,f in enumerate(feats):
        for i in np.where(f>0)[0]: arrs[i].add(f"F{fid}")
    return arrs
def token_arrays_basis(feats,labels):
    """토큰 → 켜진 발견자질의 supersense 라벨집합. 라벨은 시드간 공유 좌표."""
    arrs=[set() for _ in range(N)]
    for f,l in zip(feats,labels):
        if l is None: continue
        for i in np.where(f>0)[0]: arrs[i].add(l)
    return arrs

def avg_jaccard(arrsA,arrsB):
    js=[]
    for a,b in zip(arrsA,arrsB):
        if not a and not b: continue
        u=len(a|b)
        js.append(len(a&b)/u if u else 0.0)
    return float(np.mean(js))

# 시드쌍 간 배열 보존 — NoBasis vs Basis
import itertools
nb=[]; bs=[]
for sa,sb in itertools.combinations(SEEDS,2):
    nb.append(avg_jaccard(token_arrays_nobasis(feats_by_seed[sa]),
                          token_arrays_nobasis(feats_by_seed[sb])))
    bs.append(avg_jaccard(token_arrays_basis(feats_by_seed[sa],labels_by_seed[sa]),
                          token_arrays_basis(feats_by_seed[sb],labels_by_seed[sb])))

print("\n"+"="*60)
print("(2) 토큰 배열 시드간 보존 (Jaccard, 시드쌍 평균)")
print("="*60)
print(f"  NoBasis (발견자질ID 배열) : {np.mean(nb):.3f}")
print(f"  Basis   (supersense 라벨 배열): {np.mean(bs):.3f}")
print(f"  → Basis > NoBasis 면 기저투사가 배열 해석을 시드불변으로 단단하게.")
print(f"     향상 {np.mean(bs)-np.mean(nb):+.3f} ({np.mean(bs)/max(np.mean(nb),1e-9):.1f}배)")

# ---- (1) 라벨 안정성: 시드간 라벨 분포 일치 ----
print("\n"+"="*60)
print("(1) 라벨 안정성 — 발견자질이 받는 supersense 라벨 집합의 시드간 일치")
print("="*60)
from collections import Counter
lab_sets=[set(l for l in labels_by_seed[sd] if l) for sd in SEEDS]
inter=set.intersection(*lab_sets); uni=set.union(*lab_sets)
print(f"  시드별 라벨집합 예: {sorted(lab_sets[0])}")
print(f"  모든 시드 공통 라벨: {sorted(inter)} ({len(inter)}/{len(uni)})")
print(f"  라벨집합 Jaccard(전체): {len(inter)/len(uni):.3f}")
# 라벨 다중도(한 supersense가 몇 발견자질에 붙나) 시드간 안정?
print("\n  supersense별 부착 발견자질 수 (시드간):")
for s in sorted(uni):
    cnts=[sum(1 for l in labels_by_seed[sd] if l==s) for sd in SEEDS]
    print(f"    {s:20s} {cnts}")

json.dump(dict(nobasis_preserve=float(np.mean(nb)),basis_preserve=float(np.mean(bs)),
               label_jaccard=len(inter)/len(uni),common_labels=sorted(inter)),
          open('/home/claude/robustness_diverse.json','w'),ensure_ascii=False,indent=2)
print("\n[saved] robustness_diverse.json")
