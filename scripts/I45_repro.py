"""
I4(천장 잉여) + I5(해석 견고성) 도메인 독립 재현 — 508 legal / oncology
========================================================================
diverse에서 본 것: supersense 변별 완전잉여(FORCED 추가=DISC), 해석배열 견고성 6.8배.
508/oncology에서도 같으면 역할분리 도메인 독립.
필드 차이 흡수: 508 l1=문자열|, onco l1=리스트, (diverse ontological 이미 함).
"""
import numpy as np, json, sys
from scipy.stats import rankdata
from sklearn.cluster import KMeans
import itertools

def load(dom):
    if dom=='508':
        emb=np.load('/mnt/user-data/uploads/news_legal_embeddings.npy')
        units=json.load(open('/mnt/user-data/uploads/news_legal_units.json'))
        def ss(u):
            l=u.get('l1','')
            return l.split('|') if l else []
    elif dom=='onco':
        emb=np.load('/mnt/user-data/uploads/oncology_embeddings.npy')
        units=json.load(open('/mnt/user-data/uploads/oncology_units.json'))
        def ss(u): return u.get('l1',[]) or []
    return emb.astype(np.float64), units, [ss(u) for u in units]

def run_domain(dom):
    EMB,UNITS,tok_ss=load(dom)
    N=EMB.shape[0]
    SS=sorted({x for p in tok_ss for x in p}); ss_idx={s:j for j,s in enumerate(SS)}
    WN=np.zeros((N,len(SS)),dtype=np.int8)
    for i,parts in enumerate(tok_ss):
        for p in parts: WN[i,ss_idx[p]]=1
    base=WN.sum(0)/N

    rng=np.random.default_rng(0)
    mp=min(15000, N*(N-1)//2)
    PAIRS=rng.integers(0,N,size=(mp,2)); PAIRS=PAIRS[PAIRS[:,0]!=PAIRS[:,1]]
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

    def disc_cands(seed):
        km=KMeans(n_clusters=200,random_state=seed,n_init=3).fit(EMB)
        pr=km.cluster_centers_; pr/=(np.linalg.norm(pr,axis=1,keepdims=True)+1e-12)
        s=EMB@pr.T; Z=(s-s.mean(1,keepdims=True))/(s.std(1,keepdims=True)+1e-12)
        Bd=(Z>1.5).astype(np.int8)
        return [Bd[:,c] for c in range(200) if Bd[:,c].sum()>0]

    def greedy(cands, eps=0.001, max_feat=300, prefix=None):
        M=np.array(cands).T
        if prefix is not None:
            base_cols=np.array(prefix).T; cur=rho(base_cols)
        else:
            base_cols=np.zeros((N,0),dtype=np.int8); cur=0.0
        sel=[]; rem=list(range(M.shape[1]))
        while rem and len(sel)<max_feat:
            bc,br=None,cur
            for c in rem:
                trial=np.concatenate([base_cols,M[:,sel+[c]]],axis=1)
                r=rho(trial)
                if r>br: br,bc=r,c
            if bc is None: break
            sel.append(bc); rem.remove(bc)
            if br-cur<eps and len(sel)>=3: break
            cur=br
        return sel,cur

    # --- I4: FORCED ---
    dc=disc_cands(42)
    wn_list=[WN[:,j] for j in range(len(SS))]
    selD,rhoD=greedy(dc,max_feat=300)
    selF,rhoF=greedy(dc,max_feat=300,prefix=wn_list)
    rhoW=greedy(wn_list,max_feat=len(SS))[1]

    # --- I5: 견고성 ---
    def label_of(feat):
        on=feat.astype(bool)
        if on.sum()==0: return None
        mass=WN[on].mean(0); top=int(mass.argmax())
        if mass[top]-base[top]<0.1: return None
        return SS[top]
    def discover_feats(seed,max_feat=40):
        cands=disc_cands(seed); M=np.array(cands).T
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
    SEEDS=[42,7,123,2024]
    feats={sd:discover_feats(sd) for sd in SEEDS}
    labs={sd:[label_of(f) for f in feats[sd]] for sd in SEEDS}
    def arr_nb(fs):
        a=[set() for _ in range(N)]
        for fid,f in enumerate(fs):
            for i in np.where(f>0)[0]: a[i].add(f"F{fid}")
        return a
    def arr_bs(fs,ls):
        a=[set() for _ in range(N)]
        for f,l in zip(fs,ls):
            if l is None: continue
            for i in np.where(f>0)[0]: a[i].add(l)
        return a
    def jac(A,B):
        js=[]
        for x,y in zip(A,B):
            if not x and not y: continue
            u=len(x|y); js.append(len(x&y)/u if u else 0)
        return float(np.mean(js))
    nb=[]; bs=[]
    for sa,sb in itertools.combinations(SEEDS,2):
        nb.append(jac(arr_nb(feats[sa]),arr_nb(feats[sb])))
        bs.append(jac(arr_bs(feats[sa],labs[sa]),arr_bs(feats[sb],labs[sb])))

    return dict(N=N,n_ss=len(SS),
                I4=dict(disc=len(selD),forced_add=len(selF),diff=len(selD)-len(selF),
                        rhoD=round(rhoD,3),rhoF=round(rhoF,3),rhoW=round(rhoW,3)),
                I5=dict(nobasis=round(np.mean(nb),3),basis=round(np.mean(bs),3),
                        ratio=round(np.mean(bs)/max(np.mean(nb),1e-9),1)))

for dom in ['508','onco']:
    print("="*60); print(f"도메인: {dom}"); print("="*60)
    r=run_domain(dom)
    print(f"  N={r['N']}, supersense={r['n_ss']}")
    i4=r['I4']
    print(f"  [I4 천장잉여] DISC발견={i4['disc']}  FORCED추가={i4['forced_add']}  차이={i4['diff']}")
    print(f"             ρ: DISC {i4['rhoD']} / FORCED {i4['rhoF']} / WN단독 {i4['rhoW']}")
    print(f"             → 차이0이면 supersense 변별잉여 재현")
    i5=r['I5']
    print(f"  [I5 견고성] NoBasis={i5['nobasis']}  Basis={i5['basis']}  비율={i5['ratio']}배")
    print(f"             → Basis≫NoBasis면 해석견고성 재현 (diverse 6.8배)")
    json.dump(r,open(f'/home/claude/I45_repro_{dom}.json','w'),ensure_ascii=False,indent=2)
    print()
print("[saved] I45_repro_508.json, I45_repro_onco.json")
