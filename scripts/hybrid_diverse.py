"""
소프트 추가(A4 원안) — 워드넷 기저를 발견자질에 '추가' (교체 아님)
================================================================
이전 실패 원인: B를 A 교체로 측정(워드넷만 62자질) → ρ 0.21. 추가 아니라 갈아끼움.
교정: 발견자질(임베딩 37)은 그대로 + supersense(16)를 자질로 추가.
산술적 사실(사용자): 자질 추가 시 변별ρ는 단조 비감소. 줄면 측정오류.

세 자질집합 비교 (같은 토큰, 같은 ρ척도):
  WN_only   : supersense 16개만 (워드넷 단독 기저)
  DISC_only : 임베딩 발견자질만 (조건 A, ~37)
  HYBRID    : 발견자질 + supersense (소프트 통합 = A4 원안)
효율 입증 = 자질수 늘리며 ρ곡선: HYBRID가 같은 자질수에서 ρ 더 높고,
  목표 ρ를 더 적은 자질로 도달하면 워드넷 기저가 효율 기여.
"""
import numpy as np, json, time
from scipy.stats import rankdata
from sklearn.cluster import KMeans

EMB=np.load('/mnt/user-data/uploads/news_diverse_embeddings.npy').astype(np.float64)
UNITS=json.load(open('/mnt/user-data/uploads/news_diverse_units.json'))
N=EMB.shape[0]
tok_ss=[u.get('ontological',[]) or [] for u in UNITS]
SS=sorted({x for p in tok_ss for x in p})
ss_idx={s:j for j,s in enumerate(SS)}

# supersense 이진자질: N x 16 (멀티는 다중 활성)
WN=np.zeros((N,len(SS)),dtype=np.int8)
for i,parts in enumerate(tok_ss):
    for p in parts: WN[i,ss_idx[p]]=1

rng=np.random.default_rng(0)
PAIRS=rng.integers(0,N,size=(20000,2)); PAIRS=PAIRS[PAIRS[:,0]!=PAIRS[:,1]]
PI,PJ=PAIRS[:,0],PAIRS[:,1]
COS=np.einsum('ij,ij->i',EMB[PI],EMB[PJ])
CR=rankdata(COS); CR=CR-CR.mean(); CRN=CR/(np.linalg.norm(CR)+1e-12)

def rho(M):
    if M.shape[1]==0: return 0.0
    a=M[PI]; b=M[PJ]
    inter=(a&b).sum(1); union=(a|b).sum(1)
    with np.errstate(invalid='ignore',divide='ignore'):
        j=np.where(union>0,inter/union,0.0)
    if j.std()<1e-12: return 0.0
    jr=rankdata(j); jr=jr-jr.mean(); jn=jr/(np.linalg.norm(jr)+1e-12)
    return float(jn@CRN)

def disc_candidates(k_pool=200,z=1.5,seed=42):
    km=KMeans(n_clusters=k_pool,random_state=seed,n_init=3).fit(EMB)
    protos=km.cluster_centers_; protos/=(np.linalg.norm(protos,axis=1,keepdims=True)+1e-12)
    s=EMB@protos.T; Z=(s-s.mean(1,keepdims=True))/(s.std(1,keepdims=True)+1e-12)
    B=(Z>z).astype(np.int8)
    return [B[:,c] for c in range(k_pool) if B[:,c].sum()>0]

def greedy_curve(cands, eps=0.0005, max_feat=130):
    """후보(N차원 이진 리스트)에서 greedy ρ누적. 반환: ρ궤적(자질수별)."""
    M=np.array(cands).T
    sel=[]; cur=0.0; rem=list(range(M.shape[1])); traj=[]
    while rem and len(sel)<max_feat:
        bc,br=None,cur
        for c in rem:
            r=rho(M[:,sel+[c]])
            if r>br: br,bc=r,c
        if bc is None: break
        sel.append(bc); rem.remove(bc); traj.append(br)
        if br-cur<eps and len(sel)>=3: break
        cur=br
    return traj

SEEDS=[42,7]
t0=time.time()
print(f"diverse N={N}, supersense={len(SS)}")
print("="*64)

# WN_only는 시드무관 (supersense 고정)
wn_cands=[WN[:,j] for j in range(len(SS))]
wn_traj=greedy_curve(wn_cands, max_feat=len(SS))
print(f"WN_only (supersense 16): ρ포화 {wn_traj[-1]:.3f} @ {len(wn_traj)}자질")

curves={'DISC':[], 'HYBRID':[]}
finals={'DISC':[], 'HYBRID':[]}
widths={'DISC':[], 'HYBRID':[]}
for seed in SEEDS:
    dc=disc_candidates(seed=seed)
    # DISC_only
    td=greedy_curve(dc)
    curves['DISC'].append(td); finals['DISC'].append(td[-1]); widths['DISC'].append(len(td))
    # HYBRID: supersense를 후보에 먼저 포함 + 발견후보
    hc=[WN[:,j] for j in range(len(SS))] + dc
    th=greedy_curve(hc)
    curves['HYBRID'].append(th); finals['HYBRID'].append(th[-1]); widths['HYBRID'].append(len(th))
    print(f"  seed={seed}: DISC ρ={td[-1]:.3f}@{len(td)} | HYBRID ρ={th[-1]:.3f}@{len(th)}  [{time.time()-t0:.0f}s]")

print("\n"+"="*64)
print("최종 비교")
print("="*64)
print(f"  WN_only  : ρ {wn_traj[-1]:.3f} @ {len(wn_traj)}자질")
print(f"  DISC_only: ρ {np.mean(finals['DISC']):.3f} @ {np.mean(widths['DISC']):.0f}자질")
print(f"  HYBRID   : ρ {np.mean(finals['HYBRID']):.3f} @ {np.mean(widths['HYBRID']):.0f}자질")

# 효율: 같은 자질수(예산)에서 ρ 비교
print("\n[효율 — 같은 자질수 예산에서 ρ]")
for budget in [10,20,30,40,50]:
    dvals=[t[min(budget,len(t))-1] for t in curves['DISC']]
    hvals=[t[min(budget,len(t))-1] for t in curves['HYBRID']]
    print(f"  {budget:3d}자질: DISC ρ={np.mean(dvals):.3f}  HYBRID ρ={np.mean(hvals):.3f}  "
          f"차이 {np.mean(hvals)-np.mean(dvals):+.3f}")

print("\n[판정]")
dfin=np.mean(finals['DISC']); hfin=np.mean(finals['HYBRID'])
print(f"  HYBRID ρ({hfin:.3f}) ≥ DISC ρ({dfin:.3f})? {'예(단조)' if hfin>=dfin-0.005 else '아니오(측정오류)'}")
print(f"  → HYBRID가 같은예산서 ρ높거나 목표ρ를 적은자질로 = 워드넷기저 효율 기여")
print(f"  → 천장: HYBRID {np.mean(widths['HYBRID']):.0f} (128이내 확인)")

json.dump(dict(WN=[wn_traj[-1],len(wn_traj)],
               DISC=[float(dfin),float(np.mean(widths['DISC']))],
               HYBRID=[float(hfin),float(np.mean(widths['HYBRID']))],
               curves={k:[list(map(float,t)) for t in v] for k,v in curves.items()}),
          open('/home/claude/hybrid_diverse.json','w'),ensure_ascii=False,indent=2)
print("\n[saved] hybrid_diverse.json")
