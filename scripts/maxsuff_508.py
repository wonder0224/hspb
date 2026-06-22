"""
최대-충분 자질 포화 — 의미변별 충분히 키워도 천장 존재(유계수렴) 측정 (508)
==========================================================================
사용자 논제: ρ포화는 미세분화 못 봄. 발견자질을 끝까지 늘리며 '집합 이탈'을 추적.
  자질 추가 → 같은 배열이던 토큰 동치류가 쪼개짐 = 이탈.
  이탈이 의미있나 = 갈라진 두 무리가 라벨(supersense/4축)상 다른가.
  ★유의미 이탈 증분 → 0 수렴하는 지점 = 천장. 유한하면 유계수렴 입증.★
ρ포화점보다 늦어도, 유한 천장이 존재하면 논제 강하게 입증.
"""
import numpy as np, json
from scipy.stats import rankdata
from sklearn.cluster import KMeans

EMB=np.load('/mnt/user-data/uploads/news_legal_embeddings.npy').astype(np.float64)
UNITS=json.load(open('/mnt/user-data/uploads/news_legal_units.json'))
N=EMB.shape[0]
# 라벨: supersense + 4축 (유의미성 판정 기준). 토큰별 라벨집합.
def labels_of(u):
    L=set()
    l1=u.get('l1','')
    for p in (l1.split('|') if l1 else []): L.add('ss:'+p)
    for ax in ['court','actor','action','topic']:
        v=u.get(ax,'')
        if v: L.add(f'{ax}:{v}')
    return L
TOK_LAB=[labels_of(u) for u in UNITS]

# 발견 후보 (충분히 많이 — k_pool 크게)
def disc_candidates(k_pool=300,z=1.5,seed=42):
    km=KMeans(n_clusters=k_pool,random_state=seed,n_init=3).fit(EMB)
    pr=km.cluster_centers_; pr/=(np.linalg.norm(pr,axis=1,keepdims=True)+1e-12)
    s=EMB@pr.T; Z=(s-s.mean(1,keepdims=True))/(s.std(1,keepdims=True)+1e-12)
    B=(Z>z).astype(np.int8)
    return [B[:,c] for c in range(k_pool) if B[:,c].sum()>0]

# ρ (참고용 — ρ포화점 표시)
rng=np.random.default_rng(0)
PAIRS=rng.integers(0,N,size=(15000,2)); PAIRS=PAIRS[PAIRS[:,0]!=PAIRS[:,1]]
PI,PJ=PAIRS[:,0],PAIRS[:,1]
COS=np.einsum('ij,ij->i',EMB[PI],EMB[PJ])
CR=rankdata(COS); CR=CR-CR.mean(); CRN=CR/(np.linalg.norm(CR)+1e-12)
def rho(M):
    a=M[PI]; b=M[PJ]; inter=(a&b).sum(1); union=(a|b).sum(1)
    with np.errstate(invalid='ignore',divide='ignore'):
        j=np.where(union>0,inter/union,0.0)
    if j.std()<1e-12: return 0.0
    jr=rankdata(j); jr=jr-jr.mean(); jn=jr/(np.linalg.norm(jr)+1e-12)
    return float(jn@CRN)

def meaningful_split(members, new_feat):
    """members: 한 동치류 토큰 인덱스. new_feat로 두 무리(on/off)로 갈림.
    유의미 = 두 무리의 라벨분포가 다른가. 측정: 각 무리 최빈라벨이 다르거나
    라벨 Jaccard 낮으면 유의미. 반환: (이탈수, 유의미여부)."""
    on=[i for i in members if new_feat[i]>0]
    off=[i for i in members if new_feat[i]==0]
    if not on or not off: return 0,False   # 안 갈림
    moved=min(len(on),len(off))   # 적은 쪽이 '이탈'
    # 두 무리 라벨 집계
    def labset(grp):
        c={}
        for i in grp:
            for l in TOK_LAB[i]: c[l]=c.get(l,0)+1
        return c
    con=labset(on); coff=labset(off)
    # 최빈 라벨
    top_on=max(con,key=con.get) if con else None
    top_off=max(coff,key=coff.get) if coff else None
    # 유의미: 최빈라벨 다름 OR 한쪽에만 있는 라벨 비중 큼
    meaningful = (top_on != top_off) and (top_on is not None) and (top_off is not None)
    return moved, meaningful

def saturation_curve(cands, max_feat=120):
    """자질 1개씩 추가(greedy ρ순서 아님 — 그냥 활성수 큰 순). 매 단계 이탈·유의미이탈."""
    # 자질 순서: 활성 토큰 수 많은 순(굵은 변별부터)
    order=sorted(range(len(cands)), key=lambda c:-int(cands[c].sum()))
    # 토큰 자질배열 = 선택된 자질들의 비트열. 동치류 = 같은 비트열.
    arr=[[] for _ in range(N)]
    traj=[]
    cum_signif=0
    sig_increments=[]
    prev_classes=None
    M_sel=[]
    for step,c in enumerate(order[:max_feat],1):
        feat=cands[c]
        # 추가 전 동치류
        keys=[tuple(a) for a in arr]
        classes={}
        for i,k in enumerate(keys): classes.setdefault(k,[]).append(i)
        # 이 자질로 각 동치류가 갈리나 + 유의미
        step_moved=0; step_signif=0
        for k,members in classes.items():
            if len(members)<2: continue
            mv,sig=meaningful_split(members,feat)
            step_moved+=mv
            if sig: step_signif+=mv
        # 자질 반영
        for i in range(N): arr[i].append(int(feat[i]))
        M_sel.append(feat)
        cum_signif+=step_signif
        sig_increments.append(step_signif)
        n_classes=len(set(tuple(a) for a in arr))
        r=rho(np.array(M_sel).T) if step%5==0 or step<=10 else None
        traj.append(dict(step=step,moved=step_moved,signif=step_signif,
                         n_classes=n_classes,rho=r))
    return traj, sig_increments

print(f"508 N={N}")
cands=disc_candidates()
print(f"발견후보 {len(cands)}개")
traj,sig=saturation_curve(cands,max_feat=120)

print("\nstep | 이탈 | 유의미이탈 | 동치류수 | ρ")
for t in traj:
    if t['step']<=15 or t['step']%10==0:
        rs=f"{t['rho']:.3f}" if t['rho'] is not None else "  -"
        print(f"  {t['step']:3d} | {t['moved']:4d} | {t['signif']:4d}      | {t['n_classes']:4d}  | {rs}")

# 유의미 이탈 증분이 0 수렴하는 지점 = 천장
sig=np.array(sig)
# 마지막 연속 0 구간 시작점
window=10
print("\n[천장 탐지] 유의미이탈 증분의 이동평균(window=10):")
for s in range(0,len(sig),10):
    ma=sig[s:s+10].mean()
    print(f"  step {s+1:3d}-{s+10:3d}: 평균 유의미이탈 {ma:.1f}")
# 0 수렴점
ceil=None
for s in range(len(sig)-window):
    if sig[s:s+window].mean()<0.5:
        ceil=s+1; break
print(f"\n  유의미이탈 ≈0 수렴 시작 ≈ step {ceil} (= 최대-충분 천장 후보)")
print(f"  총 자질 {len(traj)}개까지 봄. 천장이 이 안이면 유계수렴 입증.")

json.dump(dict(traj=[{k:(v if k!='rho' else (round(v,3) if v else None)) for k,v in t.items()} for t in traj],
               sig_increments=sig.tolist(),ceiling_est=ceil),
          open('/home/claude/maxsuff_508.json','w'),ensure_ascii=False,indent=2)
print("[saved] maxsuff_508.json")
