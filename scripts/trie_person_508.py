"""
noun.person 완전 분기 트라이 — 마지막 한 순번까지 (508)
========================================================
person 토큰들을 발견자질 비트열로 트라이 구성.
자질 순번(변별순)대로 0/1 분기 → 끝까지. 마지막 한 비트만 다른 토큰이 무엇으로 갈리나.
각 분기 노드에 자질 순번 + 갈린 양쪽의 actor/topic 라벨.
유효 분기 = 갈림이 라벨상 의미 있음. 무의미 분기 = 같은 라벨 임의 쪼갬(=포화 너머).
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

# 자질 순번 = 루트(person) 안에서 변별 기여 순(루트를 균형있게 가르는 순)
def balance(idx,c):
    on=sum(1 for i in idx if Bd[i,c]>0)
    return min(on,len(idx)-on)
# person 안 활성 있는 자질만, 균형 큰 순으로 정렬 = 변별 순번
active=[c for c in range(Bd.shape[1]) if 0<sum(Bd[i,c] for i in root)<len(root)]
order=sorted(active,key=lambda c:-balance(root,c))
print(f"noun.person {len(root)}토큰, 변별 발견자질 {len(order)}개")

def lab(idx,axis):
    c=Counter()
    for i in idx:
        v=UNITS[i].get(axis,'')
        if v: c[v]+=1
    return c
def purity(idx,axis):
    d=lab(idx,axis)
    if not d: return 0,None
    t=d.most_common(1)[0]; return t[1]/sum(d.values()),t[0]

# 트라이: 자질 순번대로 재귀 분할. 의미 분기 vs 무의미 분기 기록.
log=[]
def split(idx, depth, fpos, path):
    if fpos>=len(order) or len(idx)<=1:
        log.append(('LEAF',depth,len(idx),path,
                    [UNITS[i]['text'] for i in idx[:3]],
                    purity(idx,'actor'),purity(idx,'topic')))
        return
    c=order[fpos]
    on=[i for i in idx if Bd[i,c]>0]; off=[i for i in idx if Bd[i,c]==0]
    if not on or not off:
        split(idx,depth,fpos+1,path)   # 이 자질이 안 가름 → 다음 순번
        return
    # 의미성: 양쪽 최빈 actor 다른가
    pa_on=purity(on,'actor'); pa_off=purity(off,'actor')
    meaningful = pa_on[1]!=pa_off[1] and pa_on[1] and pa_off[1]
    log.append(('SPLIT',depth,fpos,len(idx),len(on),len(off),
                pa_on,pa_off,meaningful))
    split(on,depth+1,fpos+1,path+'1')
    split(off,depth+1,fpos+1,path+'0')

import sys; sys.setrecursionlimit(10000)
split(root,0,0,'')

# 분기 요약: 깊이별 의미/무의미 분기 수
splits=[x for x in log if x[0]=='SPLIT']
leaves=[x for x in log if x[0]=='LEAF']
print(f"\n총 분기 {len(splits)}, 잎 {len(leaves)}")
from collections import defaultdict
by_depth=defaultdict(lambda:[0,0])
for x in splits:
    depth=x[1]; meaningful=x[8]
    by_depth[depth][0 if meaningful else 1]+=1
print("\n깊이별 분기 (의미 / 무의미):")
for d in sorted(by_depth):
    m,nm=by_depth[d]
    print(f"  깊이{d:2d}: 의미 {m:3d} / 무의미 {nm:3d}")

# 마지막 깊은 분기 몇 개 보기 (한 순번 차이)
print("\n=== 깊은 분기 사례 (의미 분기, 깊이 큰 것) ===")
deep=[x for x in splits if x[8]][-8:]
for x in deep:
    _,depth,fpos,ni,non,noff,pon,poff,_=x
    print(f"  깊이{depth} 자질#{order[fpos]}: n={ni}→{non}/{noff}  "
          f"ON actor={pon[1]}({pon[0]:.2f}) OFF actor={poff[1]}({poff[0]:.2f})")

print("\n=== 무의미 분기 사례 (같은 actor 임의 쪼갬 = 포화 너머) ===")
nm=[x for x in splits if not x[8]][:6]
for x in nm:
    _,depth,fpos,ni,non,noff,pon,poff,_=x
    print(f"  깊이{depth} 자질#{order[fpos]}: n={ni}→{non}/{noff}  "
          f"양쪽 actor={pon[1]}/{poff[1]} (같음=무의미)")

# 의미 분기가 깊이 어디서 멈추나
last_meaningful=max((x[1] for x in splits if x[8]),default=0)
print(f"\n→ 의미 분기 최대 깊이: {last_meaningful}")
print(f"  그 너머는 같은 actor 안 무의미 쪼갬. person 안 의미포화 깊이 = {last_meaningful}")

json.dump(dict(root=len(root),n_splits=len(splits),n_leaves=len(leaves),
               by_depth={d:by_depth[d] for d in by_depth},
               last_meaningful_depth=last_meaningful),
          open('/home/claude/trie_person_508.json','w'),ensure_ascii=False,indent=2)
print("[saved] trie_person_508.json")
