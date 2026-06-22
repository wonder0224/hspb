"""
H절 실험 — World 개체에서 발견 군집 ↔ NER 라벨 흡수율
=====================================================
(a)와의 차이: 단위가 NER 단독 개체(복합어 분리), 고유기저가 NER 라벨(PER/ORG/GPE/LOC/EVENT).
표적: (a)에서 +0.02였던 지명(GPE/LOC) 흡수가 PERSON/ORG 수준으로 오르나.
방법: F1 발견(라벨 안 봄) → 발견군집이 어느 NER라벨에 우연이상 모이나(보정 purity).
"""
import numpy as np, json
from scipy.stats import rankdata, spearmanr
from sklearn.cluster import KMeans
from collections import defaultdict, Counter

EMB = np.load('/mnt/user-data/uploads/world_entities_emb.npy').astype(np.float64)
ENTS = json.load(open('/mnt/user-data/uploads/world_entities.json'))
N = EMB.shape[0]
LABELS = ['PERSON','ORG','GPE','LOC','EVENT']
lab_idx = {l:i for i,l in enumerate(LABELS)}

# 라벨 원-핫 (개체당 단일 라벨 — NER top label)
Y = np.zeros((N, len(LABELS)))
for i,e in enumerate(ENTS):
    Y[i, lab_idx[e['ner_label']]] = 1.0
base_rate = Y.sum(0)/N
print("[setup] base rates:", {l:round(float(base_rate[lab_idx[l]]),3) for l in LABELS})

def discover(emb,k_pool=200,z=1.5,eps=0.002,max_pairs=20000,seed=42):
    rng=np.random.default_rng(seed); n=emb.shape[0]
    km=KMeans(n_clusters=k_pool,random_state=seed,n_init=3).fit(emb)
    protos=km.cluster_centers_; protos/=(np.linalg.norm(protos,axis=1,keepdims=True)+1e-12)
    sim=emb@protos.T
    Z=(sim-sim.mean(1,keepdims=True))/(sim.std(1,keepdims=True)+1e-12)
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

def absorption(B_sel):
    rows=[]
    for f in range(B_sel.shape[1]):
        on=B_sel[:,f].astype(bool)
        if on.sum()==0: continue
        mass=Y[on].mean(0)
        top=int(mass.argmax())
        rows.append(dict(n_on=int(on.sum()), top=LABELS[top],
                         raw=float(mass[top]), adj=float(mass[top]-base_rate[top])))
    return rows

print("\n[run] 5시드 발견+흡수...")
all_rows=[]
widths=[]
for seed in [42,7,123,2024,99]:
    Bs=discover(EMB,seed=seed)
    widths.append(Bs.shape[1])
    all_rows.extend(absorption(Bs))
print(f"[run] 발견폭(5시드): {widths}  평균 {np.mean(widths):.1f}")

# 라벨별 흡수 분해 — 핵심 표
print("\n"+"="*56)
print("NER 라벨별 흡수 (5시드 풀) — 표적: GPE")
print("="*56)
by=defaultdict(list); by_n=defaultdict(list)
for r in all_rows:
    by[r['top']].append(r['adj']); by_n[r['top']].append(r['n_on'])
print(f"  {'label':8s} {'기저율':>7s} {'#자질':>5s} {'평균adj':>8s} {'평균raw':>8s} {'중앙n':>6s}")
raw_by=defaultdict(list)
for r in all_rows: raw_by[r['top']].append(r['raw'])
for l in sorted(by, key=lambda x:-np.mean(by[x])):
    print(f"  {l:8s} {base_rate[lab_idx[l]]:7.3f} {len(by[l]):5d} "
          f"{np.mean(by[l]):+8.3f} {np.mean(raw_by[l]):8.3f} {int(np.median(by_n[l])):6d}")

# GPE 직접 추적: GPE를 top으로 갖는 자질 몇 개, 평균 adj
print("\n[GPE 표적 판정]")
gpe_rows=[r for r in all_rows if r['top']=='GPE']
print(f"  GPE가 top인 발견자질: {len(gpe_rows)}/{len(all_rows)}")
if gpe_rows:
    print(f"  GPE 평균 adj: {np.mean([r['adj'] for r in gpe_rows]):+.3f}  "
          f"(508 법률에선 +0.02였음)")
per=[r['adj'] for r in all_rows if r['top']=='PERSON']
org=[r['adj'] for r in all_rows if r['top']=='ORG']
print(f"  비교: PERSON {np.mean(per) if per else 0:+.3f}, ORG {np.mean(org) if org else 0:+.3f}")
print(f"  → GPE가 PER/ORG 수준이면 '깨끗한 지명은 분지된다' 입증")

json.dump(dict(widths=widths,
               by_label={l:[round(float(np.mean(by[l])),3),len(by[l]),
                            round(float(np.mean(raw_by[l])),3)] for l in by},
               base_rate={l:round(float(base_rate[lab_idx[l]]),3) for l in LABELS}),
          open('/home/claude/h_world_absorption.json','w'),ensure_ascii=False,indent=2)
print("\n[saved] h_world_absorption.json")
