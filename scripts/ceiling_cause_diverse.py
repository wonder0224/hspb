"""
천장 일치 원인 분해 — greedy 배제 vs 데이터 잉여 (diverse)
=========================================================
의문(사용자): HYBRID 천장 = DISC 천장. 코드에 천장 같게 하는 게 있나?
진단: greedy가 ρ만 보고 골라 supersense를 0개 선택 → 천장 같아짐(코드 요인).
가르기:
  시험1 FORCED: supersense 16개를 '이미 선택된' 상태로 강제 선두 → 발견자질 ρ포화까지 추가.
    천장_forced = 16 + 추가발견수.
    - 추가발견수 < DISC(86) 만큼 적으면 → supersense가 변별 일부 담당(잉여 아님). 데이터 차이.
    - 추가발견수 ≈ DISC(86) 이면 → supersense 강제로 넣어도 발견자질 그대로 필요 = 순수 잉여.
  시험2 NOCAP: max_feat 상한 키워(300) greedy 자연포화 확인 — 상한 아티팩트 배제.
  대조: supersense 선두일 때 도달 ρ vs DISC 단독 ρ. 같은 ρ를 더 적은 발견자질로 가면 기여.
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

def disc_cands(seed=42):
    km=KMeans(n_clusters=200,random_state=seed,n_init=3).fit(EMB)
    pr=km.cluster_centers_; pr/=(np.linalg.norm(pr,axis=1,keepdims=True)+1e-12)
    s=EMB@pr.T; Z=(s-s.mean(1,keepdims=True))/(s.std(1,keepdims=True)+1e-12)
    Bd=(Z>1.5).astype(np.int8)
    return [Bd[:,c] for c in range(200) if Bd[:,c].sum()>0]

def greedy(cands, eps=0.001, max_feat=300, prefix=None):
    """prefix: 강제 선두로 넣을 자질 리스트(이미 선택). 반환 sel, traj."""
    M=np.array(cands).T
    sel=[]; 
    if prefix is not None:
        P=np.array(prefix).T
        # prefix를 통째로 시작 행렬에 포함
        base_cols=P
        cur=rho(base_cols)
    else:
        base_cols=np.zeros((N,0),dtype=np.int8); cur=0.0
    rem=list(range(M.shape[1])); traj=[(len(sel),cur)]
    while rem and len(sel)<max_feat:
        bc,br=None,cur
        for c in rem:
            trial=np.concatenate([base_cols, M[:,sel+[c]]],axis=1)
            r=rho(trial)
            if r>br: br,bc=r,c
        if bc is None: break
        sel.append(bc); rem.remove(bc); traj.append((len(sel),br))
        if br-cur<eps and len(sel)>=3: break
        cur=br
    return sel, cur

SEED=42
dc=disc_cands(SEED)
wn_list=[WN[:,j] for j in range(len(SS))]
print(f"diverse N={N}, 발견후보 {len(dc)}, supersense {len(SS)}")
print("="*60)

# DISC 단독 (상한 300, 자연포화 확인)
selD,rhoD=greedy(dc, max_feat=300)
print(f"DISC 단독       : 발견 {len(selD)}개, ρ={rhoD:.3f}  (상한300서 자연포화 확인)")

# WN 단독
selW,rhoW=greedy(wn_list, max_feat=len(SS))
print(f"WN 단독         : supersense {len(selW)}개, ρ={rhoW:.3f}")

# FORCED: supersense 전체 강제 선두 + 발견 추가
selF,rhoF=greedy(dc, max_feat=300, prefix=wn_list)
print(f"FORCED(WN선두)  : supersense 16 고정 + 발견 {len(selF)}개 추가, ρ={rhoF:.3f}")
print(f"                  천장_forced = 16 + {len(selF)} = {16+len(selF)}")

print("\n"+"="*60)
print("판정 — 천장 일치는 코드냐 데이터냐")
print("="*60)
print(f"  DISC 발견수        : {len(selD)}")
print(f"  FORCED 추가발견수   : {len(selF)}")
diff=len(selD)-len(selF)
print(f"  차이(DISC - FORCED): {diff}")
if diff >= len(selD)*0.15:
    print(f"  → supersense 강제로 넣으니 발견 {diff}개 덜 필요. supersense가 변별 일부 담당.")
    print(f"    천장 일치는 greedy 배제(코드) 때문이지 순수 잉여 아님. 데이터 기여 있음.")
else:
    print(f"  → supersense 넣어도 발견 거의 그대로 필요({len(selF)}≈{len(selD)}). 순수 잉여(데이터).")
    print(f"    천장 일치는 supersense가 정말 변별 잉여라서. 코드 아티팩트 아님.")
print(f"\n  도달 ρ: DISC {rhoD:.3f} vs FORCED {rhoF:.3f} (FORCED가 높으면 supersense가 ρ 보탬)")

json.dump(dict(disc_n=len(selD),forced_add=len(selF),forced_ceiling=16+len(selF),
               rhoD=float(rhoD),rhoF=float(rhoF),rhoW=float(rhoW),diff=diff),
          open('/home/claude/ceiling_cause_diverse.json','w'),ensure_ascii=False,indent=2)
print("\n[saved] ceiling_cause_diverse.json")
