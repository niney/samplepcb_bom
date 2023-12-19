import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import json


def add_manufacturer_info(df):
    with open('manufacturers_only.json', 'r', encoding='utf-8') as f:
        manufacturers = json.load(f)
    manufacturers_set = set(manufacturers)

    def get_manufacturer(product_name):
        product_words = product_name.split()
        for word in product_words:
            if word in manufacturers_set:
                return word
        return None

    df['제조사'] = df['품명'].apply(get_manufacturer)
    return df


def split_text(text):
    pattern = re.compile(r'\[.*?\]')
    brackets = pattern.findall(text)
    text = pattern.sub('', text)
    tokens = text.split()
    tokens.extend(brackets)
    return tokens


def parse_string(text):
    classifications = {'품명': []}
    units = {
        '와트': re.compile(r"([0-9.]+/[0-9.]+)?[0-9.]*\s*([Ww])\b"),
        '오차범위': re.compile(r"[0-9.]+(\s*%)"),
        '옴': re.compile(r"([0-9.]+/[0-9.]+)?[0-9.]*\s*(ohm|OHM|Ohm|Kohm|KOHM|KOhm|MOHM|Mohm)\b"),
        '콘덴서': re.compile(r"([0-9.]+/[0-9.]+)?[0-9.]*\s*(F|f|pF|PF|pf|µF|UF|uf|mF|MF|mf)\b"),
        '전류': re.compile(r"([0-9.]+/[0-9.]+)?[0-9.]*\s*(A|a|mA|MA|ma|µA|UA|ua|nA|NA|na)\b"),
        '전압': re.compile(r"([0-9.]+/[0-9.]+)?[0-9.]*\s*(V|v|kV|KV|kv|mV|MV|mv|µV|UV|uv)\b"),
    }
    for token in text:
        classified = False
        for unit, pattern in units.items():
            if pattern.match(token):
                classifications[unit] = token
                classified = True
                break
        if not classified:
            classifications['품명'].append(token)
    return classifications


def fetch_page_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.select_one('#ajaxContents')


def fetch_and_save_manufacturers(start_page=1, end_page=1360,
                                 api_url="https://search.samplepcb.kr/api/pcbKind/_search"):
    data = []
    item_names = []

    # Iterate over the page range
    for page in range(start_page, end_page + 1):
        # Construct the API request URL
        url = f"{api_url}?page={page}&target=4"
        # Make a GET request
        response = requests.get(url)

        # If the request was successful, extract the data
        if response.status_code == 200:
            resp_data = response.json()
            data.extend(resp_data['data'])
            item_names.extend([item['itemName'] for item in resp_data['data']])

    # Save all data to a JSON file
    with open('manufacturers.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    # Save item names to a separate JSON file
    with open('manufacturers_only.json', 'w', encoding='utf-8') as f:
        json.dump(item_names, f, ensure_ascii=False, indent=4)


def main():
    base_url = 'https://eleparts.co.kr/goods/catalog?code=00020004&page={}'
    data_list = []
    for page in range(1, 2):
        url = base_url.format(page)
        items = make_item(url)
        data_list.extend(items)

    df = pd.DataFrame(data_list)
    df.to_excel('output.xlsx', index=False)


def make_item(url):
    items = []  # 페이지별로 항목을 담을 리스트
    table = fetch_page_content(url)
    rows = table.find_all('tr')

    # 제조사 정보 불러오기
    with open('manufacturers_only.json', 'r', encoding='utf-8') as f:
        manufacturers = json.load(f)
    manufacturers_set = set(manufacturers)

    for row in rows:
        title = row.find('li', {'class': 'title'})
        sub_title = row.find('li', {'class': 'subtitle'})
        price_info = row.find('span', {'class': 'boldTxt2 lnonevat'})

        title_text = title.get_text(strip=True) if title else ""
        sub_title_text = sub_title.get_text(strip=True) if sub_title else ""
        price_text = price_info.get_text(strip=True) if price_info else ""

        combined_text = title_text + " " + sub_title_text
        text = split_text(combined_text)
        classification = parse_string(text)

        product_name = ' '.join(classification.get('품명', []))
        # 제조사 정보 추가
        manufacturer = [word for word in product_name.split() if word in manufacturers_set]
        manufacturer = manufacturer[0] if manufacturer else None

        item = {
            'title': title_text,
            'subTitle': sub_title_text,
            'watt': classification.get('와트'),
            'tolerance': classification.get('오차범위'),
            'ohm': classification.get('옴'),
            'condenser': classification.get('콘덴서'),
            'current': classification.get('전류'),
            'voltage': classification.get('전압'),
            'productName': product_name,
            'price': price_text,
            'manufacturer': manufacturer  # 제조사 정보 추가
        }

        items.append(item)  # 각 항목을 리스트에 추가

    return items  # 페이지별 항목 리스트 반환

