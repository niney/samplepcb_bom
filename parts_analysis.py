import hashlib
import os
import re
from urllib.parse import urlparse, parse_qs

import requests
from bs4 import BeautifulSoup

from data_extractor import DataExtractor

extractor = DataExtractor()


class PartsAnalysis:

    def extract_param_value(self, url, param_name):
        """
        주어진 URL에서 지정된 파라미터 이름의 값을 반환합니다.

        Parameters:
        - url (str): 파라미터 값을 추출할 URL
        - param_name (str): 추출할 파라미터의 이름

        Returns:
        - str or None: 찾은 파라미터의 값 또는 해당 파라미터가 없을 경우 None
        """
        parsed_url = urlparse(url)
        params = parse_qs(parsed_url.query)

        return params.get(param_name, [None])[0]

    def fetch_page_content(self, url, cacheDir):
        # 캐시 디렉터리가 없으면 생성
        if not os.path.exists(cacheDir):
            os.mkdir(cacheDir)

        # URL을 해시값으로 변환하여 파일명으로 사용
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_filepath = os.path.join(cacheDir, url_hash)

        # 파일이 존재하면 캐싱된 내용을 사용
        if os.path.exists(cache_filepath):
            with open(cache_filepath, 'r', encoding='utf-8') as f:
                cached_html = f.read()
                return BeautifulSoup(cached_html, 'html.parser')

        # 캐싱된 파일이 없으면 웹 페이지의 내용을 가져온 후, 파일에 저장
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        content = soup.select_one('#ajaxContents')

        # 해당 내용을 파일에 저장
        with open(cache_filepath, 'w', encoding='utf-8') as f:
            f.write(str(content))

        return content

    def parse_title(self, text):
        # 제조사 정보를 []에서 추출
        pattern = re.compile(r'\[(.*?)\]')
        manufacturer_match = pattern.search(text)
        manufacturer = manufacturer_match.group(1) if manufacturer_match else None

        # 제조사 정보와 '해외' 문자열 제거하여 품명 생성
        product_name = pattern.sub('', text).replace('해외', '').strip()

        return text, product_name, manufacturer

    def parse_string(self, text):
        classifications = {'productName': []}

        units = {
            'watt': re.compile(r"([0-9.]+/[0-9.]+|[0-9.]+)\s*([Ww]|watt(s)?|WATT(S)?)\b", re.IGNORECASE),
            'errorRange': re.compile(r"[±]?[0-9.]+(\s*%)"),
            'ohm': re.compile(r"([0-9.]+/[0-9.]+)?[0-9.]*\s*(k|m)?(ohm(s)?|Ω)\b", re.IGNORECASE),
            'capacitor': re.compile(r"[0-9.]+(?:μF|µF|uF|nF|pF|mF|F)(?![a-zA-Z])", re.IGNORECASE),
            'voltage': re.compile(
                r"([0-9.]+/[0-9.]+)?[0-9.]*\s*(V|v|kV|KV|kv|mV|MV|mv|µV|UV|uv|Volt|volt|vdc|VDC|kvdc|KVDC)\b",
                re.IGNORECASE),
            'temperature': re.compile(r"(-?\d+\.?\d*)\s?(℃|°C)"),
            'size': re.compile(
                r"((\d+\.\d+|\d+)([xX*])(\d+\.\d+|\d+)(([xX*])(\d+\.\d+|\d+))?)|((\d+)(?=사이즈))|(\d+\.?\d*mm)",
                re.IGNORECASE),
            'inductor': re.compile(r"[0-9.]+(?:pH|nH|uH|mH|H)(?![a-zA-Z])", re.IGNORECASE),
            'current': re.compile(r"[0-9.]+(?:uA|µA|mA|A)(?![a-zA-Z])", re.IGNORECASE)
        }

        for token in text:
            matched_units = []

            for unit, pattern in units.items():
                match = pattern.search(token)
                if match:
                    matched_units.append(unit)
                    if unit == '온도':  # 온도는 숫자만 추출
                        classifications[unit] = match.group(1)
                    else:
                        classifications[unit] = match.group()

            if not matched_units:
                classifications['productName'].append(token)

        return classifications

    def make_item(self, url):
        items = []  # 페이지별로 항목을 담을 리스트

        code = param_value = self.extract_param_value(url, "code")
        table = self.fetch_page_content(url, "eleparts_cache_" + code)
        rows = table.find_all('tr')

        for row in rows:
            title_info = row.find('li', {'class': 'title'})
            sub_title_info = row.find('li', {'class': 'subtitle'})
            price_info = row.find('span', {'class': 'boldTxt2 lnonevat'})

            # title 파싱
            title_text, product_name, manufacturer = self.parse_title(
                title_info.get_text(strip=True)) if title_info else (
                "", "", None)
            # subTitle 파싱
            sub_title_info_get_text = sub_title_info.get_text(strip=True) if sub_title_info else ""
            sub_title_text = self.split_text(sub_title_info_get_text)
            sub_classification = self.parse_string(sub_title_text)

            # title에서 정보 파싱
            title_text_split = self.split_text(title_text)
            title_classification = self.parse_string(title_text_split)

            for key in title_classification.keys():
                if not sub_classification.get(key):  # subTitle 정보가 없는 경우만 적용
                    sub_classification[key] = title_classification[key]

            price_text = price_info.get_text(strip=True) if price_info else ""

            item = {
                'title': title_text,
                'subTitle': sub_title_info_get_text,
                'watt': sub_classification.get('와트'),
                'tolerance': sub_classification.get('오차범위'),
                'ohm': sub_classification.get('옴'),
                'condenser': sub_classification.get('콘덴서'),
                'voltage': sub_classification.get('전압'),
                'temperature': sub_classification.get('온도'),
                'size': sub_classification.get('사이즈')
                        or extractor.extract_size_from_title(sub_title_text)
                        or extractor.extract_size_from_title(title_text),
                'inductor': sub_classification.get('인덕터'),
                'current': sub_classification.get('전류'),
                'productName': product_name,
                'manufacturer': manufacturer,
                'price': price_text
            }
            # print(item)
            items.append(item)

        return items

    def split_text(self, text):
        # '[...]'를 찾아 저장하고 원래 텍스트에서 삭제
        pattern_bracket = re.compile(r'\[.*?\]')
        brackets = pattern_bracket.findall(text)
        text_without_brackets = pattern_bracket.sub('', text)

        # 숫자(소수점 포함) + 공백 + x 또는 X + 공백 + 숫자(소수점 포함) 형태의 공백을 제거하면서 반복적으로 적용
        size_pattern = re.compile(r'(\d+\.?\d*)\s*([xX]\s*\d+\.?\d*\s*)+')
        text_without_brackets = size_pattern.sub(lambda m: m.group(0).replace(' ', ''), text_without_brackets)

        # ','를 기준으로 텍스트 분할
        comma_separated = text_without_brackets.split(',')

        # '숫자 문자' 포맷의 부분에서 공백 제거
        pattern_num_char = re.compile(r'(\d+)\s+([a-zA-Z]+)')
        comma_separated = [pattern_num_char.sub(r'\1\2', segment) for segment in comma_separated]

        # 숫자 + K 혹은 M + 공백 + Ohm 형식에서 공백 제거
        ohm_pattern = re.compile(r'(\d+\.?\d*[KkMm])\s*([Oo][Hh][Mm]s?|Ω)')
        comma_separated = [ohm_pattern.sub(r'\1Ohm', segment) for segment in comma_separated]

        # 모든 세그먼트를 공백을 사용하여 합친 후 다시 공백을 기준으로 나눔
        tokens = ' '.join(comma_separated).split()

        # 저장해둔 '[...]'를 다시 추가
        tokens.extend(brackets)

        return tokens
