import json
import os
import re
import sys

from flask import Flask, jsonify
from flask import request
from flask_caching import Cache
from flask_cors import CORS
from werkzeug.utils import secure_filename
from bs4 import BeautifulSoup

import column_analysis as colanal
import column_analysis_v2 as colanal_v2
import config
import eleparts_service
import graph_ql_client
import octopart
from samplepcb_logger import make_logger

logger = None


app = Flask(__name__)

cache = Cache(app, config={
    'CACHE_TYPE': 'SimpleCache',  # 간단한 메모리 기반 캐시
    'CACHE_DEFAULT_TIMEOUT': 1800  # 캐시 유지 시간(초)
})

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
    colanal_v2.bom_ml_init_by_api()
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


@app.route('/api/v2/searchColumn', methods=['GET', 'POST'])
def search_column_v2():
    if request.method == 'POST':
        f = request.files['file']
        fname = secure_filename(f.filename)
        path = os.path.join(app.config['UPLOAD_DIR'], fname)
        f.save(path)
        query = fname
    else:
        query = request.values['q']

    ccResult = colanal.analysis_bom_pandas(query)
    return jsonify(ccResult)


@app.route('/api/analysisColumn', methods=['POST'])
def analysisColumn():
    f = request.files['file']
    fname = secure_filename(f.filename)
    path = os.path.join(app.config['UPLOAD_DIR'], fname)
    f.save(path)

    return jsonify(colanal_v2.analyzer_file(fname))

@app.route('/api/searchParts', methods=['GET', 'POST'])
@cache.cached(timeout=1800, query_string=True)
def search_parts(retry_num=1):
    client = graph_ql_client.GraphQLClient('https://octopart.com/api/v4/endpoint')
    client.inject_token(app.config['OCTOPART_APIKEY'])
    client_helper = octopart.ClientHelper()
    page = 1
    if 'page' in request.values:
        page = request.values['page']
    if 'size' in request.values:
        size = request.values['size']
    else:
        size = 2
    if 'q' in request.values:
        parts = client_helper.get_parts(client, request.values['q'], page, size, request.get_json(silent=True))
    else:
        parts = client_helper.get_parts(client, None, page, size, request.get_json(silent=True))

    if parts is None and retry_num < 5:
        logger.info('retry search parts')
        return search_parts(retry_num + 1)

    if parts['search']['hits'] > 0:
        if parts['search']['results'] is not None:
            for result in parts['search']['results']:
                part = result['part']
                is_copy_sellers_all = False
                if 'sellersAll' in request.values:
                    is_copy_sellers_all = True
                client_helper.setting_lowest_price(part, is_copy_sellers_all)
                client_helper.setting_margin(part)

    # print(parts.results)
    return jsonify(parts)


@app.route('/api/searchPartsMpn', methods=['GET', 'POST'])
@cache.cached(timeout=1800, query_string=True)
def search_parts_mpn():
    client = graph_ql_client.GraphQLClient('https://octopart.com/api/v4/endpoint')
    client.inject_token(app.config['OCTOPART_APIKEY'])
    client_helper = octopart.ClientHelper()
    parts = client_helper.get_parts_mpn(client, request.values['q'])
    if 'search_mpn' in parts and 'results' in (parts['search_mpn']) and \
            (parts['search_mpn']['results']) is not None and 'part' in (parts['search_mpn']['results'][0]):
        mpn = parts['search_mpn']['results'][0]['part']['mpn']
        if mpn == request.values['q']:
            parts['same'] = True
        else:
            parts['same'] = False
    if 'same' not in parts:
        parts['same'] = False
    return jsonify(parts)


@app.route('/api/searchLikePartsMpn', methods=['GET', 'POST'])
@cache.cached(timeout=1800, query_string=True)
def search_like_parts_mpn():
    client = graph_ql_client.GraphQLClient('https://octopart.com/api/v4/endpoint')
    client.inject_token(app.config['OCTOPART_APIKEY'])
    client_helper = octopart.ClientHelper()
    parts = client_helper.get_like_parts_mpn(client, request.values['q'])
    if 'search' in parts and 'results' in (parts['search']) and \
            (parts['search']['results']) is not None and 'part' in (parts['search']['results'][0]):
        results = parts['search']['results'][0]
        part = results['part']

        query = request.values['q']

        if part['mpn'] == request.values['q']:
            parts['same'] = True

        # Check for similarity in 'mpn' and 'name' fields
        mpn_similar = colanal_v2.similarity(part['mpn'].lower(), query.lower()) >= 0.7
        name_similar = colanal_v2.similarity(part['name'].lower(), query.lower()) >= 0.7
        # em_tag = '<em class="highlight">' in results['description']
        description_similar = False
        lower_case_word = query.lower()
        words = re.sub('<.*?>', '', results['description']).lower().split()
        for w in words:
            if len(w) >= 5 and w == lower_case_word:
                description_similar = True
        # BeautifulSoup 객체 생성
        soup = BeautifulSoup(results['description'].lower(), 'html.parser')
        # 'em' 태그와 클래스 'highlight'를 가진 요소 찾기
        soup_find = soup.find('em', class_='highlight')
        if soup_find is not None:
            highlighted_text = soup_find.text
            if highlighted_text and len(highlighted_text) >= 5 and highlighted_text == lower_case_word:
                description_similar = True

        parts['like'] = mpn_similar or name_similar or description_similar

    if 'same' not in parts:
        parts['same'] = False
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
            is_copy_sellers_all = False
            if 'sellersAll' in request.values:
                is_copy_sellers_all = True
            client_helper.setting_lowest_price(part, is_copy_sellers_all)
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


@app.route('/api/searchEleparts', methods=['GET'])
def search_eleparts():
    return jsonify(eleparts_service.make_item(request.values['url']))


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
    # colanal.bom_ml_init_by_api()
    logger.info("fit init complete!!")
    app.run(host='0.0.0.0', port=8099, debug=True)
    logger.info('stop')
