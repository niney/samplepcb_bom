#!/usr/bin/env python
# coding: utf-8

# In[1]:

import json
import logging
import random
from logging import handlers

import column_analysis as colanal
import graph_ql_client

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


def get_parts(client, q, param):

    param_str = ''
    if param is not None:
        if len(param['manufacturer_id']) != 0:
            param_str += 'manufacturer_id : [' + ','.join(param['manufacturer_id']) + ']\n'
        if len(param['mount']) != 0:
            param_str += 'mount : ["' + '","'.join(param['mount']) + '"]\n'
        if len(param['case_package']) != 0:
            param_str += 'case_package : ["' + '","'.join(param['case_package']) + '"]\n'
        if len(param['tolerance']) != 0:
            param_str += 'tolerance : ' + str(param['tolerance']) + '\n'
        if len(param['powerrating']) != 0:
            param_str += 'powerrating : ' + str(param['powerrating']) + '\n'
        if len(param['voltagerating_dc_']) != 0:
            param_str += 'voltagerating_dc_ : ' + str(param['voltagerating_dc_']) + '\n'

    filter_str = ''
    if param_str is not '':
        filter_str = """
        filters: {
          """ + param_str + """ 
        }
        """

    query = '''
    query partSearch($q: String) {
      search(
        q: $q
        country: "KR"
        currency: "KRW"
        limit: 3
        ''' + filter_str + '''
        
        ) {
        
        hits

        manufacturer_agg {
          company {
            id
            name
          }
          count
        }
        
        spec_aggs(
          attribute_names: "mount case_package tolerance powerrating voltagerating_dc_"
        ) {
          attribute {
            id
            name
            shortname
            group
          }
          buckets {
            display_value
            float_value
            count
          }
        }
                
        results {
          description
          part {
            id
            name
            short_description
            category {
              id
              name
              path
            }
            descriptions {
              text
              credit_string
              credit_url
            }
            specs {
              attribute {
                name
                shortname
                group
              }
              display_value
            }
            slug
            series {
              name
            }
            manufacturer {
              name
            }
            mpn
            generic_mpn
            best_image {
              url
              credit_url
              credit_string
            }
            images {
              url
            }
            reference_designs {
              name
              url
            }
            sellers {
              company {
                name
                aliases
                homepage_url
                slug
              }
              country
              offers {
                sku
                inventory_level
                packaging
                moq
                click_url
                # internal_url
                factory_lead_days
                factory_pack_quantity
                on_order_quantity
                multipack_quantity
                prices {
                  quantity
                  price
                  currency
                  converted_price
                  converted_currency
                  conversion_rate
                }
                order_multiple
                updated
                is_custom_pricing
              }
            }
          }
        }
      }
    }
    '''

    resp = client.execute(query, {'q': q})
    return json.loads(resp)['data']


def part_sort(x, y):
    x_offer = x['offers']
    y_offer = y['offers']
    if len(x_offer) == 0 and len(y_offer) == 0:
        return 0
    if len(x_offer) == 0 and len(y_offer) != 0:
        return 1  # y_offer 앞으로
    if len(y_offer) == 0 and len(x_offer) != 0:
        return -1  # x_offer 앞으로

    if x_offer['moq'] == 1 and y_offer['moq'] == 1:
        x_price = -1
        y_price = -1
        for price_info in x_offer['prices']:
            if price_info['quantity'] == 1:
                x_price = price_info['price']

        for price_info in y_offer['prices']:
            if price_info['quantity'] == 1:
                y_price = price_info['price']

        # 1개의 가격정보 여부 체크
        if x_price == -1 and y_offer['inventory_level'] > 0:
            return 1  # y_offer 앞으로

        if y_price == -1 and x_offer['inventory_level'] > 0:
            return -1  # x_offer 앞으로

        # 싼가격 체크
        if x_price < y_price and y_offer['inventory_level'] > 0:
            return 1  # y_offer 앞으로
        elif x_price > y_price and x_offer['inventory_level'] > 0:
            return -1  # x_offer 앞으로

    # moq none 검사
    if x_offer['moq'] is None and y_offer['moq'] is None:
        return 0
    if x_offer['moq'] is None and y_offer['moq'] is not None:
        return 1  # y_offer 앞으로
    if y_offer['moq'] is None and x_offer['moq'] is not None:
        return -1  # x_offer 앞으로

    # moq 적은 수량 우선순위
    if x_offer['moq'] > y_offer['moq']:
        return 1  # y_offer 앞으로
    if x_offer['moq'] < y_offer['moq']:
        return -1  # x_offer 앞으로

    if len(x_offer['prices']) == 0 and len(y_offer['prices']) > 0:
        return 1  # y_offer 앞으로
    if len(y_offer['prices']) == 0 and len(x_offer['prices']) > 0:
        return -1  # x_offer 앞으로
    if len(y_offer['prices']) == 0 and len(x_offer['prices']) == 0:
        return 0

    x_price = x_offer['prices'][0]['price']
    y_price = y_offer['prices'][0]['price']

    # 재고체크
    if x_price < y_price and y_offer['inventory_level'] > 0:
        return 1  # y_offer 앞으로
    elif x_price > y_price and x_offer['inventory_level'] > 0:
        return -1  # x_offer 앞으로

    return 0

##### MAIN SCRIPT #####


import os
import sys
import config

from flask import Flask, jsonify
from flask import request
from flask_cors import CORS
from werkzeug.utils import secure_filename
from functools import cmp_to_key

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
    parts = get_parts(client, request.values['q'], request.get_json())
    if parts['search']['hits'] > 0:
        for result in parts['search']['results']:
            for i, seller in enumerate(result['part']['sellers']):
                cut_tape_offer = None
                for offer in seller['offers']:
                    if offer['packaging'] == 'Cut Tape':
                        cut_tape_offer = offer

                if cut_tape_offer is None:
                    offers_list = result['part']['sellers'][i]['offers']
                    if len(offers_list) == 1:
                        random_idx = 0
                    else:
                        random_idx = random.randrange(0, len(offers_list) - 1)
                    result['part']['sellers'][i]['offers'] = result['part']['sellers'][i]['offers'][random_idx]
                else:
                    result['part']['sellers'][i]['offers'] = cut_tape_offer

            searched_sellers = sorted(result['part']['sellers'], key=cmp_to_key(part_sort))
            if len(searched_sellers) > 0:
                result['part']['sellers'] = searched_sellers[0]

    # print(parts.results)
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
