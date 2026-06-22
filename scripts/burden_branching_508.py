"""
부담 분배 동역학 검증 (508)
============================
사용자 해석: 고유기저가 거칠수록 군집이 변별부담을 더 떠안는다(자기부담 증가).
직접 측정: 한 고유라벨당 발견군집이 평균 몇 갈래로 정렬되나(분기수).
  거침(4라벨) → 한 라벨에 여러 군집 몰림(분기 큼) = 군집이 많이 보충
  잘음(62라벨) → 한 라벨당 군집 적음(분기 작음) = 군집 부담 적음
예측: 분기수가 거침>중간>잘음 단조. = 부담분배 동역학.

분기수 정의: 각 발견군집을 그 최빈 고유라벨에 귀속 → 라벨별 군집 수 →
  (정렬된 군집 총수) / (실제 정렬받은 라벨 수) = 라벨당 평균 분기.
또 하나: 라벨당 분기 분포(어떤 라벨이 많이 갈리나).
"""
import numpy as np, json
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from collections import defaultdict, Counter

EMB=np.load('/mnt/user-data/uploads/news_legal_embeddings.npy').astype(np.float64)
UNITS=json.load(open('/mnt/user-data/uploads/news_legal_units.json'))
N=EMB.shape[0]

all_ss=[]; tok_ss=[]
for u in UNITS:
    parts=u['l1'].split('|') if u['l1'] else []
    tok_ss.append(parts); all_ss.extend(parts)
SS=sorted(set(all_ss)); ss_idx={s:j for j,s in enumerate(SS)}
COARSE_MAP={
 'noun.person':'ENTITY','noun.group':'ENTITY','noun.location':'ENTITY',
 'noun.object':'ENTITY','noun.artifact':'ENTITY','noun.substance':'ENTITY','noun.body':'ENTITY',
 'noun.event':'EVENTIVE','noun.act':'EVENTIVE','noun.state':'EVENTIVE',
 'noun.process':'EVENTIVE','noun.phenomenon':'EVENTIVE',
 'noun.communication':'INFO','noun.cognition':'INFO','noun.attribute':'INFO',
 'noun.time':'OTHER'}
COARSE=sorted(set(COARSE_MAP.values())); c_idx={c:j for j,c in enumerate(COARSE)}
AXES=['court','actor','action','topic']
FINE=sorted({f"{ax}:{u[ax]}" for u in UNITS for ax in AXES if u.get(ax,'')})
f_idx={l:j for j,l in enumerate(FINE)}

def build_Y(level):
    if level=='mid':
        Y=np.zeros((N,len(SS)))
        for i,parts in enumerate(tok_ss):
            if parts:
                w=1.0/len(parts)
                for p in parts: Y[i,ss_idx[p]]+=w
        return Y,SS
    if level=='coarse':
        Y=np.zeros((N,len(COARSE)))
        for i,parts in enumerate(tok_ss):
            if parts:
                w=1.0/len(parts)
                for p in parts: Y[i,c_idx[COARSE_MAP[p]]]+=w
        return Y,COARSE
    if level=='fine':
        Y=np.zeros((N,len(FINE)))
        for i,u in enumerate(UNITS):
            labs=[f"{ax}:{u[ax]}" for ax in AXES if u.get(ax,'')]
            if labs:
                w=1.0/len(labs)
                for l in labs: Y[i,f_idx[l]]+=w
        return Y,FINE

def discover(emb,k_pool=200,z=1.5,eps=0.002,max_pairs=15000,seed=42):
    rng=np.random.default_rng(seed); n=emb.shape[0]
    km=KMeans(n_clusters=k_pool,random_state=seed,n_init=3).fit(emb)
    protos=km.cluster_centers_; protos/=(np.linalg.norm(protos,axis=1,keepdims=True)+1e-12)
    s=emb@protos.T; Z=(s-s.mean(1,keepdims=True))/(s.std(1,keepdims=True)+1e-12)
    B=(Z>z).astype(np.int8)
    pairs=rng.integers(0,n,size=(max_pairs,2)); pairs=pairs[pairs[:,0]!=pairs[:,1]]
    pi,pj=pairs[:,0],pairs[:,1]
    cos=np.einsum('ij,ij->i',emb[pi],emb[pj])
    cr=rankdata(cos); cr=cr-cr.mean(); crn=cr/(np.linalg.norm(cr)+1e-12)
    def rho(cols):
        Bs=B[:,cols]; a=Bs[pi]; b=Bs[pj]
        inter=(a&b).sum(1); union=(a|b).sum(1)
        with np.errstate(invalid='ignore',divide='ignore'):
            j=np.where(union>0,inter/union,0.0)
        if j.std()<1e-12: return -1.0
        jr=rankdata(j); jr=jr-jr.mean(); jn=jr/(np.linalg.norm(jr)+1e-12)
        return float(jn@crn)
    sel=[]; cur=0.0; rem=[c for c in range(k_pool) if B[:,c].sum()>0]
    while rem:
        bc,br=None,cur
        for c in rem:
            r=rho(sel+[c])
            if r>br: br,bc=r,c
        if bc is None: break
        g=br-cur; sel.append(bc); rem.remove(bc)
        if g<eps and len(sel)>=3: break
        cur=br
    return B[:,sel]

def branching(B_sel,Y):
    """각 발견군집을 최빈 고유라벨에 귀속 → 라벨당 군집 수."""
    lab_of=[]
    for f in range(B_sel.shape[1]):
        on=B_sel[:,f].astype(bool)
        if on.sum()==0: continue
        mass=Y[on].mean(0)
        if mass.sum()<1e-9: continue
        lab_of.append(int(mass.argmax()))
    cnt=Counter(lab_of)
    n_clusters=len(lab_of)
    n_labels_used=len(cnt)
    avg_branch=n_clusters/max(n_labels_used,1)
    return avg_branch,n_clusters,n_labels_used,cnt

print("="*60)
print("부담 분배 — 입도별 한 고유라벨당 군집 분기수 (5시드 평균)")
print("="*60)
res={}
for lv in ['coarse','mid','fine']:
    Y,names=build_Y(lv)
    branches=[]; ncs=[]; nls=[]
    pooled_cnt=Counter()
    for seed in [42,7,123,2024,99]:
        Bs=discover(EMB,seed=seed)
        ab,nc,nl,cnt=branching(Bs,Y)
        branches.append(ab); ncs.append(nc); nls.append(nl)
        for k,v in cnt.items(): pooled_cnt[names[k]]+=v
    res[lv]={'avg_branch':float(np.mean(branches)),'n_labels_total':len(names),
             'n_labels_used':float(np.mean(nls)),'n_clusters':float(np.mean(ncs))}
    print(f"  {lv:7s} ({len(names):2d}라벨): 라벨당 분기 {np.mean(branches):.2f}  "
          f"(군집 {np.mean(ncs):.0f}개 → 정렬라벨 {np.mean(nls):.0f}개)")
    top=pooled_cnt.most_common(4)
    print(f"           최다 분기 라벨: " + ", ".join(f"{k}({v})" for k,v in top))

print("\n[해석]")
b_c,b_m,b_f=res['coarse']['avg_branch'],res['mid']['avg_branch'],res['fine']['avg_branch']
print(f"  분기수: 거침 {b_c:.2f} → 중간 {b_m:.2f} → 잘음 {b_f:.2f}")
if b_c>b_m>b_f:
    print("  → 단조 증가(거침일수록 분기 큼). 사용자 해석 입증:")
    print("     고유기저 거칠수록 군집이 한 범주를 여러 갈래로 보충 = 자기부담 증가.")
else:
    print("  → 단조 아님. 부담분배 해석 재검토 필요.")
print(f"  발견 군집 총수는 입도 무관 ~{res['mid']['n_clusters']:.0f}개 고정(유계).")
print(f"  → 같은 유계 군집이, 거친 기저에선 적은 라벨에 몰리고(분기↑),")
print(f"     잘은 기저에선 많은 라벨에 분산(분기↓). 총량은 천장 안에서 재분배.")

json.dump(res,open('/home/claude/burden_branching_508.json','w'),ensure_ascii=False,indent=2)
print("\n[saved] burden_branching_508.json")
