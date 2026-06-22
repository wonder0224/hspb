"""
noun.person 분기 — 5시드 평균 (정량값 본문 인용용)
====================================================
extract_person_branch.py를 5시드로 돌려 분기수·의미분기수·최대깊이·잎수의
평균±std 산출. 그림은 시드42 raw로, 본문 수치는 이 평균으로.

경로: PC에서 C:\\CL_token_experiment 기준으로 EMB/UNITS 경로만 바꿔 실행.
(샌드박스에서도 /mnt/user-data/uploads 로 동작)
"""
import numpy as np, json
from sklearn.cluster import KMeans
from collections import Counter, defaultdict

# --- 경로 (PC: 아래 두 줄을 C:\CL_token_experiment\... 로) ---
EMB_PATH='/mnt/user-data/uploads/news_legal_embeddings.npy'
UNITS_PATH='/mnt/user-data/uploads/news_legal_units.json'

EMB=np.load(EMB_PATH).astype(np.float64)
UNITS=json.load(open(UNITS_PATH))
N=EMB.shape[0]
def ss_of(u):
    l=u.get('l1',''); return l.split('|') if l else []
root=[i for i in range(N) if 'noun.person' in ss_of(UNITS[i])]

def purity(idx,axis):
    c=Counter()
    for i in idx:
        v=UNITS[i].get(axis,'')
        if v: c[v]+=1
    if not c: return 0.0,None
    t=c.most_common(1)[0]; return t[1]/sum(c.values()),t[0]

def run_seed(seed):
    km=KMeans(n_clusters=200,random_state=seed,n_init=3).fit(EMB)
    pr=km.cluster_centers_; pr/=(np.linalg.norm(pr,axis=1,keepdims=True)+1e-12)
    s=EMB@pr.T; Z=(s-s.mean(1,keepdims=True))/(s.std(1,keepdims=True)+1e-12)
    Bd=(Z>1.5).astype(np.int8)
    def balance(idx,c):
        on=sum(1 for i in idx if Bd[i,c]>0); return min(on,len(idx)-on)
    active=[c for c in range(Bd.shape[1]) if 0<sum(Bd[i,c] for i in root)<len(root)]
    order=sorted(active,key=lambda c:-balance(root,c))
    stat=dict(n_split=0,n_meaningful=0,n_leaf=0,max_md=0,n_feat=len(order))
    def split(idx,depth,fpos):
        if fpos>=len(order) or len(idx)<=1:
            stat['n_leaf']+=1; return
        c=order[fpos]
        on=[i for i in idx if Bd[i,c]>0]; off=[i for i in idx if Bd[i,c]==0]
        if not on or not off: split(idx,depth,fpos+1); return
        pon=purity(on,'actor'); poff=purity(off,'actor')
        meaningful=bool(pon[1] and poff[1] and pon[1]!=poff[1])
        stat['n_split']+=1
        if meaningful:
            stat['n_meaningful']+=1; stat['max_md']=max(stat['max_md'],depth)
        split(on,depth+1,fpos+1); split(off,depth+1,fpos+1)
    import sys; sys.setrecursionlimit(10000)
    split(root,0,0)
    return stat

SEEDS=[42,7,123,2024,99]
rows=[run_seed(s) for s in SEEDS]
keys=['n_feat','n_split','n_meaningful','n_leaf','max_md']
print(f"noun.person root = {len(root)} tokens, {len(SEEDS)} seeds")
print(f"{'metric':14s} {'mean':>8s} {'std':>7s}   per-seed")
summary={}
for k in keys:
    v=np.array([r[k] for r in rows],float)
    summary[k]=dict(mean=round(v.mean(),2),std=round(v.std(),2),
                    vals=[int(x) for x in v])
    print(f"{k:14s} {v.mean():8.2f} {v.std():7.2f}   {[int(x) for x in v]}")

json.dump(dict(root_size=len(root),seeds=SEEDS,summary=summary),
          open('person_branch_5seed.json','w'),ensure_ascii=False,indent=2)
print("\n[saved] person_branch_5seed.json")
print("→ 본문 정량값은 이 mean±std 사용. 그림은 시드42 raw(구조 예시).")
