import json
import logging
import random

from functools import cmp_to_key


class ClientHelper:
    samplepcb_auth_provider = ['mouser', 'digi-key', 'element14-apac']

    def part_sort(self, x, y):
        # 공급업체
        x_company = x['company']
        y_company = y['company']
        if len(x_company) == 0 and len(y_company) != 0:
            return 1  # y_company 앞으로
        if len(y_company) == 0 and len(x_company) != 0:
            return -1  # x_company 앞으로
        if len(x_company) != 0 and len(y_company) != 0:
            if not (y_company['slug'] in self.samplepcb_auth_provider and x_company['slug'] in self.samplepcb_auth_provider):
                if y_company['slug'] in self.samplepcb_auth_provider:
                    return 1
                if x_company['slug'] in self.samplepcb_auth_provider:
                    return -1

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
                    x_price = price_info['converted_price']

            for price_info in y_offer['prices']:
                if price_info['quantity'] == 1:
                    y_price = price_info['converted_price']

            # 1개의 가격정보 여부 체크
            if x_price == -1 and y_offer['inventory_level'] > 0:
                return 1  # y_offer 앞으로

            if y_price == -1 and x_offer['inventory_level'] > 0:
                return -1  # x_offer 앞으로

            # 싼가격 체크
            if x_price < y_price and y_offer['inventory_level'] > 0:
                return -1  # x_offer 앞으로
            elif x_price > y_price and x_offer['inventory_level'] > 0:
                return 1  # y_offer 앞으로

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

    def setting_lowest_price(self, part, is_copy_sellers_all=False):
        for i, seller in enumerate(part['sellers']):
            cut_tape_offer = None
            for offer in seller['offers']:
                if offer['packaging'] == 'Cut Tape':
                    cut_tape_offer = offer

            if cut_tape_offer is None:
                offers_list = part['sellers'][i]['offers']
                if len(offers_list) == 1:
                    random_idx = 0
                else:
                    random_idx = random.randrange(0, len(offers_list) - 1)
                part['sellers'][i]['offers'] = part['sellers'][i]['offers'][random_idx]
            else:
                part['sellers'][i]['offers'] = cut_tape_offer
        searched_sellers = sorted(part['sellers'], key=cmp_to_key(self.part_sort))
        if len(searched_sellers) > 0:
            part['sellers'] = searched_sellers[0]
        if is_copy_sellers_all:
            part['sellersAll'] = searched_sellers

    def setting_margin(self, part):
        """
        가격에 마진을 적용
        seller 는 복사본을 쓰기 때문에 한군데만 적용시켜준다
        :param part:
        :return:
        """
        if 'sellersAll' in part:
            for seller in part['sellersAll']:
                self.setting_margin_by_seller(seller)
        else:
            searched_seller = part['sellers']
            self.setting_margin_by_seller(searched_seller)

    def setting_margin_by_seller(self, seller):
        if 'offers' in seller and 'prices' in seller['offers']:
            for price in seller['offers']['prices']:
                price['converted_price'] = price['converted_price'] + (price['converted_price'] * 0.02)
                price['marginRate'] = 2

    def get_parts_query(self):
        """
        octopart parts 쿼리 가져오기
        :return:
        """
        return '''
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
            best_datasheet {
              name
              url
              created_at
            }
            images {
              url
            }
            reference_designs {
              name
              url
            }
            sellers(
              authorized_only: true
            ) {
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
        '''

    def get_parts(self, client, q, page=1, param=None):
        filter_param_str = ''
        q_str = ''
        categories_str = ''
        if q is not None:
            q_str = 'q: "' + q + '"\n'

        page_str = 'start: ' + str((int(page) - 1) * 10) + '\n'

        if param is not None:
            if 'manufacturer_id' in param and len(param['manufacturer_id']) != 0:
                filter_param_str += 'manufacturer_id : [' + ','.join(param['manufacturer_id']) + ']\n'
            # if 'mount' in param and len(param['mount']) != 0:
            #     filter_param_str += 'mount : ["' + '","'.join(param['mount']) + '"]\n'
            # if 'case_package' in param and len(param['case_package']) != 0:
            #     filter_param_str += 'case_package : ["' + '","'.join(param['case_package']) + '"]\n'
            # if 'tolerance' in param and len(param['tolerance']) != 0:
            #     filter_param_str += 'tolerance : ' + str(param['tolerance']) + '\n'
            # if 'powerrating' in param and len(param['powerrating']) != 0:
            #     filter_param_str += 'powerrating : ' + str(param['powerrating']) + '\n'
            # if 'voltagerating_dc_' in param and len(param['voltagerating_dc_']) != 0:
            #     filter_param_str += 'voltagerating_dc_ : ' + str(param['voltagerating_dc_']) + '\n'
            if 'filters' in param:
                for f_key, f_val in param['filters'].items():
                    if len(f_val) != 0:
                        filter_param_str += f'{f_key} : ["' + '","'.join(f_val) + '"]\n'
            if 'categories' in param:
                if 'ids' in param['categories']:
                    categories_str += 'ids : ["' + '","'.join(param['categories']['ids']) + '"]\n'
                    filter_param_str += 'category_id : ["' + '","'.join(param['categories']['ids']) + '"]\n'

        filter_str = ''
        if filter_param_str is not '':
            filter_str = """
            filters: {
              """ + filter_param_str + """ 
            }
            """

        if categories_str is not '':
            categories_str = """
                categories(
                    """ + categories_str + """
                  ) {
                    id
                    name
                    path
                    parent_id
                    ancestors {
                      id
                      name
                      children {
                        id
                        name
                      }
                    }
                    children {
                      id
                      name
                      path
                      children {
                        id
                        name
                        path
                      }
                    }
                  }
            """

        spec_aggs_names_str = ''
        if param is not None and 'specAggsNames' in param:
            spec_aggs_names_str = " ".join(param['specAggsNames'])

        query = '''
            query partSearch {
            
              ''' + categories_str + '''
            
              search(
                ''' + q_str + '''
                country: "KR"
                currency: "KRW"
                ''' + page_str + '''
                limit: 10
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
                  attribute_names: "'''+ spec_aggs_names_str+'''"
                  size: 255
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
                
                category_agg {
                  category {
                    parent_id
                    id
                    name
                    path
                  }
                  count
                }
                        
                results {
                  description
                  part {
                  ''' + self.get_parts_query() + '''
                  }
                }
              }
            }
        '''

        resp = client.execute(query)
        return json.loads(resp)['data']

    def get_parts_by_ids(self, client, param=None):

        query = '''
            query partIdSearch($ids: [String!]!) {
                parts(
                    ids: $ids
                    country: "KR"
                    currency: "KRW"
                ) {
                    ''' + self.get_parts_query() + '''
                }
            }
        '''
        resp = client.execute(query, param)
        return json.loads(resp)['data']

    def get_categories(self, client, param=None):

        categories_str = ''
        if 'categories' in param:
            if 'ids' in param['categories']:
                categories_str += 'ids : ["' + '","'.join(param['categories']['ids']) + '"]\n'

        categories_str = """
            categories(
                """ + categories_str + """
              ) {
                id
                name
                path
                parent_id
                ancestors {
                  id
                  name
                  children {
                    id
                    name
                  }
                }
                children {
                  id
                  name
                  path
                  children {
                    id
                    name
                    path
                  }
                }
              }
        """

        query = '''
            query partSearch {
              ''' + categories_str + '''
            }
        '''

        resp = client.execute(query)
        return json.loads(resp)['data']

    def get_manufacturers(self, client, param=None):
        manufacturers_str = '''
            manufacturers {
              id
              name
              aliases
              display_flag
              homepage_url
              slug
              is_verified
              is_distributorapi
            }
        '''

        query = '''
            query partSearch {
              ''' + manufacturers_str + '''
            }
        '''

        resp = client.execute(query)
        return json.loads(resp)['data']

    def get_suggested_filters(self, client, q, param=None):

        q_str = ''
        if q is not None:
            q_str = 'q: "' + q + '"\n'

        filter_param_str = ''
        if param is not None and 'categories' in param:
            if 'ids' in param['categories']:
                filter_param_str += 'category_id : ["' + '","'.join(param['categories']['ids']) + '"]\n'

        filter_str = ''
        if filter_param_str is not '':
            filter_str = """
            filters: {
              """ + filter_param_str + """ 
            }
            """

        query = '''
          query suggestedSearch {
            search(
               ''' + q_str + '''
               ''' + filter_str + '''
            ) {
              suggested_filters {
                id
                name
                shortname
                group
              }
            } 
          }
        '''
        resp = client.execute(query)
        return json.loads(resp)['data']
