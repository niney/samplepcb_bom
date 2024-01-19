#!/usr/bin/env python
# coding: utf-8

# In[39]:


import logging
from logging import handlers

import numpy as np
from flask_caching import Cache

import column_analysis as colanal
import sentence_hub as sh

logger = None


# In[ ]:


def make_logger(name=None):
    # 1 logger instance를 만든다.
    logger = logging.getLogger(name)

    # 2 logger의 level을 가장 낮은 수준인 DEBUG로 설정해둔다.
    logger.setLevel(logging.DEBUG)

    # 3 formatter 지정
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # 4 handler instance 생성
    console = logging.StreamHandler()
    file_handler = handlers.TimedRotatingFileHandler(filename='app.log', when='midnight', interval=1, encoding='utf-8')

    # 5 handler 별로 다른 level 설정
    console.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)

    # 6 handler 출력 format 지정
    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    file_handler.suffix = "%Y%m%d"

    # 7 logger에 handler 추가
    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger


# In[50]:


##### MAIN SCRIPT #####

from flask import Flask, jsonify
from flask import request

app = Flask(__name__)

cache = Cache(app, config={
    'CACHE_TYPE': 'SimpleCache',  # 간단한 메모리 기반 캐시
    'CACHE_DEFAULT_TIMEOUT': 1800  # 캐시 유지 시간(초)
})

@app.route('/doc2vect', methods=['POST'])
def req_doc2vect():
    query = request.form['q']
    vect = sh.doc2vect(query)
    # vectList = ' '.join(str(x) for x in vect)
    ccResult = {'result': True, 'data': vect[0]}
    logger.info(ccResult)
    return jsonify(ccResult)


def generate_cache_key_by_qlist():
    # 'q[]' 파라미터를 기반으로 리스트를 얻습니다.
    query_list = request.form.getlist('q[]')

    # 리스트를 정렬하여 순서가 달라도 동일한 내용이면 같은 키를 생성하도록 합니다.
    query_list.sort()

    # 리스트를 문자열로 변환하여 캐시 키로 사용합니다.
    return str(query_list)


def generate_cache_key_by_target():
    # 'target'과 'targetList[]' 파라미터 값을 얻습니다.
    target = request.form['target']
    target_list = request.form.getlist('targetList[]')

    # 리스트를 정렬하여 순서가 달라도 동일한 내용이면 같은 키를 생성하도록 합니다.
    target_list.sort()

    # 'target'과 정렬된 'targetList'를 결합하여 캐시 키로 사용합니다.
    return f"{target}_{str(target_list)}"


@app.route('/docs2vects', methods=['POST'])
@cache.cached(timeout=50, key_prefix=generate_cache_key_by_qlist)
def req_docs2vects():
    query = request.form.getlist('q[]')
    vects = sh.docs2vects(query)
    ccResult = {'result': True, 'data': vects}
    return jsonify(ccResult)


@app.route('/embedScore', methods=['POST'])
@cache.cached(timeout=50, key_prefix=generate_cache_key_by_target)
def req_embed_score():
    target = request.form['target']
    query = request.form.getlist('targetList[]')
    target_embed = sh.model(target)
    query_embed = sh.model(query)
    data = np.inner(target_embed, query_embed)
    ccResult = {'result': True, 'data': data.tolist()[0]}
    return jsonify(ccResult)


@app.route('/searchColumn', methods=['GET'])
def search_column():
    query = request.values['q']
    ccResult = colanal.analysis_bom(query)
    return jsonify(ccResult)


if __name__ == '__main__':
    # try:
    logger = make_logger()
    logger.info("Downloading pre-trained embeddings from tensorflow hub...")
    sh.init()
    logger.info("init complete!!!!!!")

    app.run(host='0.0.0.0', port=8096)
    logger.info('stop')
