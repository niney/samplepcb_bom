#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging
from logging import handlers

import column_analysis as colanal

logger = None


# In[2]:


def make_logger(name=None):
    # 1 logger instance를 만든다.
    logger = logging.getLogger(name)

    # 2 logger의 level을 가장 낮은 수준인 DEBUG로 설정해둔다.
    logger.setLevel(logging.DEBUG)

    # 3 formatter 지정
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # 4 handler instance 생성
    console = logging.StreamHandler()
    file_handler = handlers.TimedRotatingFileHandler(filename='app_prototype.log', when='midnight', interval=1,
                                                     encoding='utf-8')

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


# In[9]:


##### MAIN SCRIPT #####


import os
import sys
import config

from flask import Flask, jsonify
from flask import request
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)

# cors
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.route('/', methods=['GET', 'POST'])
def index():
    return "running"


@app.route('/api', methods=['GET', 'POST'])
def indexApi():
    return "running"


@app.route('/api/init', methods=['GET'])
def init():
    colanal.bom_ml_init_by_api()
    return jsonify({'result': True})


@app.route('/api/searchColumn', methods=['GET', 'POST'])
def search_column():
    # 'PartList_DRSD-Atype A TOP Rev02A r02.xlsx'
    if request.method == 'POST':
        f = request.files['file']
        fname = secure_filename(f.filename)
        path = os.path.join(app.config['UPLOAD_DIR'], fname)
        f.save(path)
        query = fname
    else:
        query = request.values['q']

    ccResult = colanal.analysis_bom(query)
    return jsonify(ccResult)


if __name__ == '__main__':
    # try:
    env = sys.argv[1] if len(sys.argv) > 1 else 'dev'

    if env == 'dev':
        app.config.from_object(config.DevelopmentConfig)
    elif env == 'prod':
        app.config.from_object(config.ProductionConfig)
    else:
        app.config.from_object(config.Config)

    logger = make_logger()
    # colanal.bom_ml_init()  # ml init
    colanal.bom_ml_init_by_api()
    logger.info("fit init complete!!")
    app.run(host='0.0.0.0', port=8099)
    logger.info('stop')
