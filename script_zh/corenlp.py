import os
os.environ['CORENLP_HOME'] = "/data/home/donglz/codespace/AIGT/stanford-corenlp-4.5.7"
from stanza.server import CoreNLPClient

# 示例数据
data = [
    {"article": "近来，美国等西方国家以安全原因为由发布针对土耳其的安全预警或接连关闭在土耳其的领事馆。对此，土耳其内政部长索伊卢3日在安塔利亚指责美国驻土耳其大使杰弗里·弗莱克试图搅乱土耳其民众视线，制造混乱。"}
]

with CoreNLPClient(
        annotators=['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref'],
        endpoint="http://localhost:9099",
        properties='chinese',
        memory='10G') as client:
    for idx, line in enumerate(data):
        text = line["article"]
        ann = client.annotate(text)

        mychains = list()
        chains = ann.corefChain
        for chain in chains:
            mychain = list()
            for mention in chain.mention:
                words_list = ann.sentence[mention.sentenceIndex].token[mention.beginIndex:mention.endIndex]
                #build a string out of the words of this mention
                ment_word = ' '.join([x.word for x in words_list])
                mychain.append(ment_word)
            mychains.append(mychain)

        for chain in mychains:
            print(' <-> '.join(chain))