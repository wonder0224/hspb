"""
noun.person 루트 토큰별 분기 raw 추출 — 그림용
============================================
각 person 토큰이 trie에서 어느 경로로 갈리는지, 공유 자질 수(depth),
최종 잎 크기, 4축 라벨을 토큰 단위로 출력. 집계가 아니라 raw.
출력: person_branch_raw.json (그림 그리는 새 채팅이 실제 토큰으로 nested 원 구성)
"""
import numpy as np, json
from sklearn.cluster import KMeans
from collections import Counter

EMB=np.load('/mnt/user-data/uploads/news_legal_embeddings.npy').astype(np.float64)
UNITS=json.load(open('/mnt/user-data/uploads/news_legal_units.json'))
N=EMB.shape[0]
def ss_of(u):
    l=u.get('l1',''); return l.split('|') if l else []
root=[i for i in range(N) if 'noun.person' in ss_of(UNITS[i])]

km=KMeans(n_clusters=200,random_state=42,n_init=3).fit(EMB)
pr=km.cluster_centers_; pr/=(np.linalg.norm(pr,axis=1,keepdims=True)+1e-12)
s=EMB@pr.T; Z=(s-s.mean(1,keepdims=True))/(s.std(1,keepdims=True)+1e-12)
Bd=(Z>1.5).astype(np.int8)

def balance(idx,c):
    on=sum(1 for i in idx if Bd[i,c]>0)
    return min(on,len(idx)-on)
active=[c for c in range(Bd.shape[1]) if 0<sum(Bd[i,c] for i in root)<len(root)]
order=sorted(active,key=lambda c:-balance(root,c))

def lab(idx,axis):
    c=Counter()
    for i in idx:
        v=UNITS[i].get(axis,'')
        if v: c[v]+=1
    return c
def purity(idx,axis):
    d=lab(idx,axis)
    if not d: return 0.0,None
    t=d.most_common(1)[0]; return t[1]/sum(d.values()),t[0]

# 토큰별: 분기 경로(0/1 문자열), 공유자질수(=경로에서 실제 가른 자질 수), 최종 잎 동료수
token_path={i:'' for i in root}
token_depth={i:0 for i in root}

# 노드 기록: (depth, 자질순번, size, on/off size, actor purity 양쪽, 의미여부, 멤버 일부)
nodes=[]

def split(idx, depth, fpos, path):
    if fpos>=len(order) or len(idx)<=1:
        for i in idx:
            token_path[i]=path; token_depth[i]=depth
        pa=purity(idx,'actor'); pt=purity(idx,'topic')
        nodes.append(dict(kind='leaf',depth=depth,size=len(idx),path=path,
                          actor=pa[1],actor_pur=round(pa[0],3),
                          topic=pt[1],topic_pur=round(pt[0],3),
                          members=[UNITS[i]['text'] for i in idx[:6]]))
        return
    c=order[fpos]
    on=[i for i in idx if Bd[i,c]>0]; off=[i for i in idx if Bd[i,c]==0]
    if not on or not off:
        split(idx,depth,fpos+1,path); return
    pon=purity(on,'actor'); poff=purity(off,'actor')
    meaningful = bool(pon[1] and poff[1] and pon[1]!=poff[1])
    nodes.append(dict(kind='split',depth=depth,feat=int(c),size=len(idx),
                      on=len(on),off=len(off),
                      on_actor=pon[1],on_pur=round(pon[0],3),
                      off_actor=poff[1],off_pur=round(poff[0],3),
                      meaningful=meaningful,path=path))
    split(on,depth+1,fpos+1,path+'1')
    split(off,depth+1,fpos+1,path+'0')

import sys; sys.setrecursionlimit(10000)
split(root,0,0,'')

# 토큰 단위 raw 테이블
tokens=[]
for i in root:
    tokens.append(dict(idx=int(i),text=UNITS[i]['text'],
                       depth=token_depth[i],path=token_path[i],
                       actor=UNITS[i].get('actor',''),topic=UNITS[i].get('topic',''),
                       n_active=int(Bd[i].sum())))

# depth별 집계(검증용 — 새 채팅의 purity_by_depth와 대조)
from collections import defaultdict
band=defaultdict(list)
def bandof(d):
    if d<=1: return '0-1'
    if d<=3: return '2-3'
    if d<=5: return '4-5'
    if d<=8: return '6-8'
    return '9+'
leaves=[n for n in nodes if n['kind']=='leaf']
for n in leaves:
    band[bandof(n['depth'])].append((n['size'],n['actor_pur']))
band_summary={}
for b,v in band.items():
    sizes=[x[0] for x in v]; purs=[x[1] for x in v]
    band_summary[b]=dict(n_leaves=len(v),mean_size=round(np.mean(sizes),1),
                         mean_purity=round(np.mean(purs),3))

# 자질 0개 토큰 = 단일 분화 토큰 (사용자 지적)
zero_feat=[UNITS[i]['text'] for i in root if Bd[i].sum()==0]

out=dict(
    root_size=len(root),
    n_discriminative_features=len(order),
    n_splits=len([n for n in nodes if n['kind']=='split']),
    n_meaningful_splits=len([n for n in nodes if n['kind']=='split' and n['meaningful']]),
    n_leaves=len(leaves),
    last_meaningful_depth=max((n['depth'] for n in nodes if n['kind']=='split' and n['meaningful']),default=0),
    band_summary=band_summary,
    zero_feature_tokens=dict(count=len(zero_feat),examples=zero_feat[:20]),
    nodes=nodes,
    tokens=tokens,
)
json.dump(out,open('/home/claude/person_branch_raw.json','w'),ensure_ascii=False,indent=2)
print(f"root(person)={len(root)}, 변별자질={len(order)}, 분기={out['n_splits']}(의미 {out['n_meaningful_splits']}), 잎={len(leaves)}")
print(f"의미분기 최대깊이={out['last_meaningful_depth']}")
print("band_summary:", json.dumps(band_summary,ensure_ascii=False))
print(f"자질0개 토큰(단일분화)={len(zero_feat)}개:", zero_feat[:10])
print("[saved] person_branch_raw.json")
