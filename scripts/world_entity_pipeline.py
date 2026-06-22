"""
world_entity_pipeline.py  —  [당신 PC에서 실행]
================================================
목적: AG News World(label 0) 문서에서 표준 NER로 개체를 추출(단독 명사=개체),
      PER/ORG/GPE/LOC/EVENT 라벨을 붙이고, 각 distinct 개체를 임베딩.
      → 다음 단계(발견+흡수율)는 Claude 환경에서.

왜 이렇게: 기존 단위가 gpt-4o 복합명사구(`federal court in Washington D.C.`)라
  LOC(지명)가 ORG에 묻혔다. NER로 단독 개체(`Washington D.C.` 분리)만 뽑아
  깨끗한 고유 기저(NER 라벨)를 만든다. NER은 결정적·우리가설 독립 → 동어반복 회피.

준비:
  pip install spacy openai numpy
  python -m spacy download en_core_web_sm   (또는 정확도 원하면 en_core_web_trf)
  set OPENAI_API_KEY=...

출력:
  world_entities.json   : [{entity_id, text, ner_label, doc_count}, ...]  (distinct)
  world_entities_emb.npy: (n_entities x 3072) float32, json과 행 정렬
  → 이 둘을 Claude에 업로드.

설계 고정(LEXICON G/H절):
- distinct 개체 단위(같은 'Russia'는 1회 임베딩). 라벨 충돌 시 최빈.
- 핵심 5축: PERSON, ORG, GPE, LOC, EVENT.  나머지 = MISC.
  GPE(국가/도시/주)와 LOC(산/강/비행정)는 합치지 않고 따로 — '지명' 흡수를 각각 본다.
- 표적: GPE/LOC 흡수가 (a)의 +0.02에서 PERSON/ORG 수준(+0.2~0.4)으로 오르나.
"""
import os, csv, sys, json, time
import numpy as np

CSV_PATH   = r"C:\CL_token_experiment\data\diverse\ag_news_sample_4000.csv"
WORLD_LABEL = "0"                         # AG News: 0=World
KEEP = {"PERSON","ORG","GPE","LOC","EVENT"}   # 핵심 축, 나머지는 MISC
MIN_LEN = 2                               # 개체 텍스트 최소 길이
SPACY_MODEL = os.environ.get("SPACY_MODEL","en_core_web_sm")  # 정확도면 en_core_web_trf
EMB_MODEL  = "text-embedding-3-large"
OUT_JSON   = r"C:\CL_token_experiment\data\diverse\world_entities.json"
OUT_NPY    = r"C:\CL_token_experiment\data\diverse\world_entities_emb.npy"
DO_EMBED   = os.environ.get("DO_EMBED","1") == "1"   # 0이면 NER 통계만 보고 멈춤

csv.field_size_limit(min(sys.maxsize, 2**31-1))

def extract_entities():
    import spacy
    nlp = spacy.load(SPACY_MODEL, disable=["lemmatizer"])
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8-sig")))
    world = [r["text"] for r in rows if r["label"] == WORLD_LABEL]
    print(f"[ner] World 문서 {len(world)}개 처리 (model={SPACY_MODEL})")
    # distinct 개체: text -> {labels Counter, count}
    from collections import Counter, defaultdict
    bag = defaultdict(lambda: {"labels": Counter(), "count": 0})
    for i, doc in enumerate(nlp.pipe(world, batch_size=64)):
        for ent in doc.ents:
            lab = ent.label_ if ent.label_ in KEEP else "MISC"
            if lab == "MISC":
                continue   # 개체만 분석 — MISC(날짜/수량 등)는 버림
            t = ent.text.strip()
            if len(t) < MIN_LEN:
                continue
            bag[t]["labels"][lab] += 1
            bag[t]["count"] += 1
        if (i+1) % 200 == 0:
            print(f"  ...{i+1} docs, distinct so far={len(bag)}")
    entities = []
    for eid,(t,info) in enumerate(sorted(bag.items())):
        top_label = info["labels"].most_common(1)[0][0]
        entities.append({"entity_id": eid, "text": t,
                         "ner_label": top_label, "doc_count": info["count"],
                         "label_dist": dict(info["labels"])})
    # 라벨 분포 보고
    lc = Counter(e["ner_label"] for e in entities)
    print(f"[ner] distinct 개체 {len(entities)}개")
    for lab,c in lc.most_common():
        print(f"     {lab:8s} {c}")
    return entities

def embed_entities(entities):
    from openai import OpenAI
    client = OpenAI()
    texts = [e["text"] for e in entities]
    vecs = []
    B = 256
    for i in range(0, len(texts), B):
        chunk = texts[i:i+B]
        r = client.embeddings.create(model=EMB_MODEL, input=chunk)
        vecs.extend([d.embedding for d in r.data])
        print(f"  embedded {min(i+B,len(texts))}/{len(texts)}")
        time.sleep(0.2)
    arr = np.asarray(vecs, dtype=np.float32)
    # L2 정규화(508과 동일 규약)
    arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
    return arr

def main():
    entities = extract_entities()
    json.dump(entities, open(OUT_JSON,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[out] {OUT_JSON}")
    # GPE/LOC 표본 충분한지 경고
    from collections import Counter
    lc = Counter(e["ner_label"] for e in entities)
    if lc.get("GPE",0) + lc.get("LOC",0) < 30:
        print(f"[warn] 지명(GPE+LOC) {lc.get('GPE',0)+lc.get('LOC',0)}개 — 표본 작음. "
              f"코퍼스 확대 고려.")
    if not DO_EMBED:
        print("[stop] DO_EMBED=0 — NER 통계만. 임베딩 생략."); return
    arr = embed_entities(entities)
    np.save(OUT_NPY, arr)
    print(f"[out] {OUT_NPY}  shape={arr.shape}")
    print("\n두 파일(world_entities.json, world_entities_emb.npy)을 Claude에 업로드하세요.")

if __name__ == "__main__":
    main()
