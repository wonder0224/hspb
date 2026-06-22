"""
HYBRID에서 supersense가 실제 선택되나 + 워드넷의 진짜 값어치
============================================================
첫 실험: HYBRID ρ = DISC ρ (동일). 추정: greedy가 supersense를 0개 골랐다.
확인: HYBRID greedy 선택 목록에서 supersense vs 발견자질 비율.
함의: 0개면 워드넷은 변별 잉여(발견자질이 다 포함). 값어치는 해석층.
해석 검증: 발견자질 각각이 어느 supersense와 정렬되나(라벨 부착 가능성).
빠르게: 1시드, max_feat 줄여서.
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
PAIRS=rng.integers(0,N,size=(20000,2)); PAIRS=PAIRS[PAIRS[:,0]!=PAIRS[:,1]]
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

km=KMeans(n_clusters=200,random_state=42,n_init=3).fit(EMB)
protos=km.cluster_centers_; protos/=(np.linalg.norm(protos,axis=1,keepdims=True)+1e-12)
s=EMB@protos.T; Z=(s-s.mean(1,keepdims=True))/(s.std(1,keepdims=True)+1e-12)
Bd=(Z>1.5).astype(np.int8)
disc=[Bd[:,c] for c in range(200) if Bd[:,c].sum()>0]
print(f"발견후보 {len(disc)}개, supersense {len(SS)}개")

# HYBRID greedy: 후보 = supersense 16 (인덱스 0~15) + 발견 (16~)
cands=[WN[:,j] for j in range(len(SS))] + disc
TYPE=['WN']*len(SS)+['DISC']*len(disc)
M=np.array(cands).T
sel=[]; cur=0.0; rem=list(range(M.shape[1])); order=[]
while rem and len(sel)<60:
    bc,br=None,cur
    for c in rem:
        r=rho(M[:,sel+[c]])
        if r>br: br,bc=r,c
    if bc is None: break
    sel.append(bc); rem.remove(bc); order.append((TYPE[bc],br))
    if br-cur<0.0005 and len(sel)>=3: break
    cur=br

n_wn=sum(1 for t,_ in order if t=='WN')
n_disc=sum(1 for t,_ in order if t=='DISC')
print(f"\nHYBRID 선택 {len(order)}개 중: WN(supersense) {n_wn}개, DISC(발견) {n_disc}개")
print("선택 순서(앞 20):", " ".join(t[0] for t,_ in order[:20]))
print(f"최종 ρ = {order[-1][1]:.3f}")
if n_wn==0:
    print("→ supersense 0개 선택. 워드넷은 변별 잉여(발견자질이 다 포함).")
else:
    print(f"→ supersense {n_wn}개 선택됨. 발견자질이 못 잡는 변별을 워드넷이 보충.")
    wn_picked=[SS[ (cands.index(WN[:,j])) ] for j in range(len(SS)) if any(sel[k]==j for k in range(len(sel)))]

# 해석층: 발견자질 각각이 어느 supersense와 정렬(부착 라벨)
print("\n[해석층] 발견자질 → 최빈 supersense (사후 라벨 부착 가능성)")
disc_sel=[c for c in sel if TYPE[c]=='DISC'][:12]
for c in disc_sel:
    on=M[:,c].astype(bool)
    if on.sum()==0: continue
    mass=WN[on].mean(0); top=int(mass.argmax())
    adj=mass[top]-base[top]
    print(f"  발견자질(n={on.sum():4d}) → {SS[top]:20s} adj={adj:+.3f}")

json.dump(dict(n_wn=n_wn,n_disc=n_disc,final_rho=order[-1][1],
               order=[t for t,_ in order]),
          open('/home/claude/hybrid_select_diverse.json','w'),ensure_ascii=False,indent=2)
print("\n[saved]")
