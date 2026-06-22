"""
oncology 부담 분배 동역학 (508 결과 교차검증)
=============================================
508에서 입증: 거친 고유기저일수록 군집 분기↑, 군집 총수는 입도무관 고정(유계 재분배).
oncology(4629토큰, state/substance/process 중심)에서 같은 패턴인가?
입도: 거침(supersense→4상위) / 중간(supersense 16) / 잘음(4축 pathology/part/domain 펼침)
"""
import numpy as np, json, time
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from collections import defaultdict, Counter

EMB=np.load('/mnt/user-data/uploads/oncology_embeddings.npy').astype(np.float64)
UNITS=json.load(open('/mnt/user-data/uploads/oncology_units.json'))
N=EMB.shape[0]

# supersense (리스트형)
all_ss=[]; tok_ss=[]
for u in UNITS:
    parts=u.get('l1',[]) or []
    tok_ss.append(parts); all_ss.extend(parts)
SS=sorted(set(all_ss)); ss_idx={s:j for j,s in enumerate(SS)}

# 거침: 508과 동일한 상위묶음 규칙
COARSE_MAP={
 'noun.person':'ENTITY','noun.group':'ENTITY','noun.location':'ENTITY',
 'noun.object':'ENTITY','noun.artifact':'ENTITY','noun.substance':'ENTITY','noun.body':'ENTITY',
 'noun.event':'EVENTIVE','noun.act':'EVENTIVE','noun.state':'EVENTIVE',
 'noun.process':'EVENTIVE','noun.phenomenon':'EVENTIVE',
 'noun.communication':'INFO','noun.cognition':'INFO','noun.attribute':'INFO',
 'noun.time':'OTHER'}
COARSE=sorted(set(COARSE_MAP.values())); c_idx={c:j for j,c in enumerate(COARSE)}

# 잘음: oncology 4축 (pathology/part/domain — entity_type 제외, 508의 court류 대응)
AXES=['pathology','part','domain']
FINE=sorted({f"{ax}:{v}" for u in UNITS for ax in AXES for v in (u.get(ax) or [])})
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
            labs=[f"{ax}:{v}" for ax in AXES for v in (u.get(ax) or [])]
            if labs:
                w=1.0/len(labs)
                for l in labs: Y[i,f_idx[l]]+=w
        return Y,FINE

def discover(emb,k_pool=200,z=1.5,eps=0.002,max_pairs=20000,seed=42):
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
    lab_of=[]
    for f in range(B_sel.shape[1]):
        on=B_sel[:,f].astype(bool)
        if on.sum()==0: continue
        mass=Y[on].mean(0)
        if mass.sum()<1e-9: continue
        lab_of.append(int(mass.argmax()))
    cnt=Counter(lab_of)
    return len(lab_of)/max(len(cnt),1), len(lab_of), len(cnt), cnt

SEEDS=[42,7,123]   # 4629라 3시드
print(f"oncology N={N}, supersense={len(SS)}, fine라벨={len(FINE)}")
print("="*60)
print("부담 분배 — 입도별 분기수 (발견은 시드당 1회, 세 입도 재사용)")
print("="*60)
t0=time.time()

# 발견을 시드당 1회 (입도 무관) — 미리 다 계산
discovered={}
for seed in SEEDS:
    discovered[seed]=discover(EMB,seed=seed)
    print(f"  [discover seed={seed}] 군집 {discovered[seed].shape[1]}개  [{time.time()-t0:.0f}s]")

Ys={lv:build_Y(lv) for lv in ['coarse','mid','fine']}
res={}
for lv in ['coarse','mid','fine']:
    Y,names=Ys[lv]
    branches=[]; ncs=[]; nls=[]; pooled=Counter()
    for seed in SEEDS:
        Bs=discovered[seed]
        ab,nc,nl,cnt=branching(Bs,Y)
        branches.append(ab); ncs.append(nc); nls.append(nl)
        for k,v in cnt.items(): pooled[names[k]]+=v
    res[lv]={'avg_branch':float(np.mean(branches)),'n_labels_total':len(names),
             'n_labels_used':float(np.mean(nls)),'n_clusters':float(np.mean(ncs))}
    print(f"  {lv:7s} ({len(names):3d}라벨): 분기 {np.mean(branches):.2f}  "
          f"(군집 {np.mean(ncs):.0f} → 정렬라벨 {np.mean(nls):.1f})")
    print(f"           최다분기: " + ", ".join(f"{k}({v})" for k,v in pooled.most_common(4)))

print("\n[해석]")
b_c,b_m,b_f=res['coarse']['avg_branch'],res['mid']['avg_branch'],res['fine']['avg_branch']
print(f"  분기수: 거침 {b_c:.2f} → 중간 {b_m:.2f} → 잘음 {b_f:.2f}")
print(f"  군집 총수: {[round(res[l]['n_clusters'],0) for l in ['coarse','mid','fine']]} (입도무관 고정이면 유계 재분배)")
mono = b_c>b_m>b_f
print(f"  단조감소(거침>중간>잘음): {mono}")
print(f"  → 508과 같은 패턴이면 부담보존 동역학 도메인 독립.")
print(f"\n  [508 대조] 508: 분기 12.8→6.6→3.7, 군집 26 고정")

json.dump(res,open('/home/claude/burden_branching_oncology.json','w'),ensure_ascii=False,indent=2)
print("[saved] burden_branching_oncology.json")
