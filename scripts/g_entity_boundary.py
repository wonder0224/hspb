"""
entity vs 술어 경계 검증
========================
가설(사용자): 발견 군집은 전통 NER 범주(person/location/organization=group)와
  event를 잘 분지(흡수)하고, 술어성 noun(act/state/cognition)과는 직교.
검증: 발견자질을 top_ss로 가르지 말고, 각 자질이 entity묶음 vs 술어묶음에
  얼마나 정렬되는지 양쪽 다 본다. location을 명시 추적.

entity 묶음 = noun.person, noun.location, noun.group, noun.event, noun.artifact, noun.object
술어 묶음   = noun.act, noun.state, noun.cognition, noun.attribute, noun.process,
              noun.communication, noun.phenomenon, noun.time, noun.substance, noun.body
(communication/time은 경계적 — 별도 표기)
"""
import numpy as np, json
from scipy.stats import rankdata, mannwhitneyu
from sklearn.cluster import KMeans
from collections import defaultdict

EMB = np.load('/mnt/user-data/uploads/news_legal_embeddings.npy').astype(np.float64)
UNITS = json.load(open('/mnt/user-data/uploads/news_legal_units.json'))
N=EMB.shape[0]

all_ss=[]; tok_ss=[]
for u in UNITS:
    parts=u['l1'].split('|') if u['l1'] else []
    tok_ss.append(parts); all_ss.extend(parts)
SS=sorted(set(all_ss)); ss_idx={s:j for j,s in enumerate(SS)}
S=np.zeros((N,len(SS)))
for i,parts in enumerate(tok_ss):
    if parts:
        w=1.0/len(parts)
        for p in parts: S[i,ss_idx[p]]+=w
base_rate=S.sum(0)/N

ENTITY = {"noun.person","noun.location","noun.group","noun.event","noun.artifact","noun.object"}
PRED   = {"noun.act","noun.state","noun.cognition","noun.attribute","noun.process","noun.phenomenon"}
# communication, time, substance, body 는 중립으로 둠

def discover(emb,k_pool=200,z=1.5,eps=0.002,max_pairs=15000,seed=42):
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

# 발견자질별: entity묶음 adj vs 술어묶음 adj, 그리고 top entity supersense
def analyze(B_sel):
    rows=[]
    for f in range(B_sel.shape[1]):
        on=B_sel[:,f].astype(bool)
        if on.sum()==0: continue
        mass=S[on].mean(0)
        # entity 묶음 총질량 - 기저, 술어 묶음 총질량 - 기저
        ent_mass=sum(mass[ss_idx[s]] for s in ENTITY if s in ss_idx)
        ent_base=sum(base_rate[ss_idx[s]] for s in ENTITY if s in ss_idx)
        prd_mass=sum(mass[ss_idx[s]] for s in PRED if s in ss_idx)
        prd_base=sum(base_rate[ss_idx[s]] for s in PRED if s in ss_idx)
        # 자질이 가장 정렬된 entity supersense
        ent_scores={s: mass[ss_idx[s]]-base_rate[ss_idx[s]] for s in ENTITY if s in ss_idx}
        top_ent=max(ent_scores,key=ent_scores.get)
        rows.append(dict(n_on=int(on.sum()),
                         ent_adj=float(ent_mass-ent_base),
                         prd_adj=float(prd_mass-prd_base),
                         top_ent=top_ent, top_ent_adj=float(ent_scores[top_ent])))
    return rows

print("="*64)
print("entity 묶음 vs 술어 묶음 정렬 (5시드 풀)")
print("="*64)
all_rows=[]
for seed in [42,7,123,2024,99]:
    all_rows.extend(analyze(discover(EMB,seed=seed)))

ent=np.array([r['ent_adj'] for r in all_rows])
prd=np.array([r['prd_adj'] for r in all_rows])
print(f"  발견자질 총 {len(all_rows)}개")
print(f"  entity 묶음 평균 adj: {ent.mean():+.3f}  (양수=entity로 모임)")
print(f"  술어   묶음 평균 adj: {prd.mean():+.3f}  (음수=술어 회피)")
print(f"  자질별 entity-술어 차이 평균: {(ent-prd).mean():+.3f}")
u,p=mannwhitneyu(ent,prd,alternative='greater')
print(f"  Mann-Whitney(entity>술어) p={p:.2e}")
n_ent_dom=sum(1 for r in all_rows if r['ent_adj']>r['prd_adj'])
print(f"  entity-우세 자질: {n_ent_dom}/{len(all_rows)} = {n_ent_dom/len(all_rows):.2f}")

print("\n"+"="*64)
print("각 발견자질이 정렬된 top entity supersense 분포")
print("="*64)
top_ent_count=defaultdict(list)
for r in all_rows:
    top_ent_count[r['top_ent']].append(r['top_ent_adj'])
print(f"  {'entity supersense':22s} {'기저율':>7s} {'#자질':>5s} {'평균adj':>8s}")
for s in sorted(top_ent_count,key=lambda x:-np.mean(top_ent_count[x])):
    br = base_rate[ss_idx[s]] if s in ss_idx else 0
    print(f"  {s:22s} {br:7.3f} {len(top_ent_count[s]):5d} {np.mean(top_ent_count[s]):+8.3f}")

# location 명시 확인
print("\n[location 추적]")
loc_present = 'noun.location' in ss_idx
if loc_present:
    print(f"  noun.location 기저율: {base_rate[ss_idx['noun.location']]:.3f} (토큰 18/508)")
    loc_aligned=[r for r in all_rows if r['top_ent']=='noun.location']
    print(f"  location이 top-entity인 발견자질: {len(loc_aligned)}개")
    if loc_aligned:
        print(f"  그 자질들 location adj 평균: {np.mean([r['top_ent_adj'] for r in loc_aligned]):+.3f}")

json.dump(dict(ent_mean=float(ent.mean()),prd_mean=float(prd.mean()),
               diff=float((ent-prd).mean()),mwu_p=float(p),
               ent_dominant_frac=n_ent_dom/len(all_rows),
               top_ent={s:[round(float(np.mean(v)),3),len(v)] for s,v in top_ent_count.items()}),
          open('/home/claude/g_entity_boundary.json','w'),ensure_ascii=False,indent=2)
print("\n[saved] g_entity_boundary.json")
