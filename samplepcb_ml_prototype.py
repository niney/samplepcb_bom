#!/usr/bin/env python
# coding: utf-8

# In[1]:

import json
import logging
from logging import handlers

import column_analysis as colanal
import graph_ql_client
import octopart

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

def get_parts2(client, ids):
    query = '''
    query get_parts($ids: [String!]!) {
        parts(ids: $ids) {
            id
            manufacturer {
                name
            }
            mpn
            category {
                name
            }
        }
    }
    '''

    ids = [str(id) for id in ids]
    resp = client.execute(query, {'ids': ids})
    return json.loads(resp)['data']['parts']


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


@app.route('/api/searchParts', methods=['GET', 'POST'])
def search_parts():
    client = graph_ql_client.GraphQLClient('https://octopart.com/api/v4/endpoint')
    client.inject_token(app.config['OCTOPART_APIKEY'])
    client_helper = octopart.ClientHelper()
    page = 1
    if 'page' in request.values:
        page = request.values['page']
    if 'q' in request.values:
        parts = client_helper.get_parts(client, request.values['q'], page, request.get_json())
    else:
        parts = client_helper.get_parts(client, None, page, request.get_json())

    if parts['search']['hits'] > 0:
        if parts['search']['results'] is not None:
            for result in parts['search']['results']:
                part = result['part']
                if 'sellersAll' in request.values:
                    part['sellersAll'] = part['sellers']
                client_helper.setting_lowest_price(part)
                client_helper.setting_margin(part)

    # print(parts.results)
    return jsonify(parts)


@app.route('/api/searchCategories', methods=['GET', 'POST'])
def search_categories():
    client = graph_ql_client.GraphQLClient('https://octopart.com/api/v4/endpoint')
    client.inject_token(app.config['OCTOPART_APIKEY'])

    client_helper = octopart.ClientHelper()
    parts = client_helper.get_categories(client, request.get_json())

    return jsonify(parts)


@app.route('/api/searchPartsByIds', methods=['GET', 'POST'])
def search_parts_by_ids():
    client = graph_ql_client.GraphQLClient('https://octopart.com/api/v4/endpoint')
    client.inject_token(app.config['OCTOPART_APIKEY'])
    client_helper = octopart.ClientHelper()
    resp = client_helper.get_parts_by_ids(client, request.get_json())

    if len(resp['parts']) > 0:
        for part in resp['parts']:
            if 'sellersAll' in request.values:
                part['sellersAll'] = part['sellers']
            client_helper.setting_lowest_price(part)
            client_helper.setting_margin(part)
    return jsonify(resp)


@app.route('/api/v1/searchQuery', methods=['GET', 'POST'])
def search_query():
    client = graph_ql_client.GraphQLClient('https://octopart.com/api/v4/endpoint')
    client.inject_token(app.config['OCTOPART_APIKEY'])
    resp = client.execute(request.get_json()['query'], request.get_json()['variables'])
    return json.loads(resp)['data']


@app.route('/api/searchManufacturers', methods=['GET', 'POST'])
def search_manufacturers():
    client = graph_ql_client.GraphQLClient('https://octopart.com/api/v4/endpoint')
    client.inject_token(app.config['OCTOPART_APIKEY'])

    client_helper = octopart.ClientHelper()
    parts = client_helper.get_manufacturers(client, request.get_json())

    return jsonify(parts)


@app.route('/api/searchSuggested', methods=['GET', 'POST'])
def search_suggested():
    client = graph_ql_client.GraphQLClient('https://octopart.com/api/v4/endpoint')
    client.inject_token(app.config['OCTOPART_APIKEY'])
    client_helper = octopart.ClientHelper()
    if len(request.data) == 0:
        json_data = None
    else:
        json_data = request.get_json()
    if 'q' in request.values:
        parts = client_helper.get_suggested_filters(client, request.values['q'], json_data)
    else:
        parts = client_helper.get_suggested_filters(client, None, json_data)
    return jsonify(parts)


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
