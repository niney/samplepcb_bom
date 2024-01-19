import re
import requests

from data_extractor import DataExtractor

extractor = DataExtractor()


class PartsAnalysis:

    def parse_title(self, text):
        # 제조사 정보를 []에서 추출
        pattern = re.compile(r'\[(.*?)\]')
        manufacturer_match = pattern.search(text)
        manufacturer = manufacturer_match.group(1) if manufacturer_match else None

        # 제조사 정보와 '해외' 문자열 제거하여 품명 생성
        product_name = pattern.sub('', text).replace('해외', '').strip()

        return text, product_name, manufacturer

    def parse_string(self, text, reference_prefix=None):
        classifications = {'productName': []}

        units = {
            'watt': re.compile(r"([0-9.]+/[0-9.]+|[0-9.]+)\s*([Ww]|watt(s)?|WATT(S)?)\b", re.IGNORECASE),
            'errorRange': re.compile(r"[±]?[0-9.]+(\s*%)"),
            'ohm': re.compile(r"([0-9.]+/[0-9.]+)?[0-9.]*\s*(k|m)?(ohm(s)?|Ω)\b", re.IGNORECASE),
            'parrot': re.compile(r"(?<!\S)[0-9.]+(?:μF|µF|uF|nF|pF|mF|F)(?!\S)", re.IGNORECASE),
            'voltage': re.compile(
                # r"([0-9.]+/[0-9.]+)?[0-9.]*\s*(V|v|kV|KV|kv|mV|MV|mv|µV|UV|uv|Volt|volt|vdc|VDC|kvdc|KVDC)\b",
                r"\b[0-9.]+\s*(V|v|kV|KV|kv|mV|MV|mv|µV|UV|uv|Volt|volt|vdc|VDC|kvdc|KVDC)\b",
                re.IGNORECASE),
            'temperature': re.compile(r"(-?\d+\.?\d*)\s?(℃|°C)"),
            'size': re.compile(
                r"(?<!\S)(\d+\.\d+|\d+)(([xX*](\d+\.\d+|\d+))?([xX*](\d+\.\d+|\d+))?(\s*(mm|사이즈))?)(?=\s|$)",
                re.IGNORECASE),
            # r"(\d+\.\d+|\d+)([xX*])(\d+\.\d+|\d+)(?!\S)(\s*|$)(([xX*])(\d+\.\d+|\d+)(?!\S)(\s*|$))?",
            # r"((\d+\.\d+|\d+)([xX*])(\d+\.\d+|\d+)(([xX*])(\d+\.\d+|\d+))?)|((\d+)(?=사이즈))|(\d+\.?\d*mm)",
            # re.IGNORECASE),
            'henry': re.compile(r"(?<!\S)[0-9.]+(?:pH|nH|uH|mH|H)(?!\S)(?=\s|$)", re.IGNORECASE),
            'current': re.compile(r"(?<!\S)[0-9.]+(?:uA|µA|mA|A)\b", re.IGNORECASE),
            'frequency': re.compile(r"[0-9.]+(?:Hz|kHz|MHz|GHz)\b", re.IGNORECASE),
        }

        for token in text:
            matched_units = []

            for unit, pattern in units.items():
                match = pattern.search(token)
                if match:
                    if unit == 'temperature':  # 온도는 숫자만 추출
                        classifications[unit] = match.group(1)
                        matched_units.append(unit)
                    if unit == 'size':
                        size = match.group()
                        if not size.isdigit() and not self.is_float(size):  # size 값이 숫자가 아닌 경우에만 저장
                            classifications[unit] = size
                            matched_units.append(unit)
                    else:
                        classifications[unit] = match.group()
                        matched_units.append(unit)

            # referencePrefix를 사용하여 특정 분류에 append_text_to_specific_format 실행
            if reference_prefix and reference_prefix in ['C', 'R', 'L']:
                unit_mapping = {'C': 'parrot', 'R': 'ohm', 'L': 'henry'}
                target_unit = unit_mapping.get(reference_prefix)
                if target_unit not in classifications:
                    specific_format_result = self.append_text_to_specific_format(token, reference_prefix)
                    if specific_format_result:
                        matched_units.append(specific_format_result)
                        classifications[target_unit] = specific_format_result

            if not matched_units:
                classifications['productName'].append(token)

        # 'size' 키가 classifications에 없는 경우에만 실행
        if 'size' not in classifications:
            size = extractor.extract_size_from_title(text)  # size 값을 추출
            # 추출된 size가 None이 아닌 경우에만 classifications에 추가
            if size is not None:
                classifications['size'] = size
                classifications['productName'] = [name for name in classifications['productName'] if name != size]

        return classifications

    def split_text(self, text):
        # 'μ'와 'µ'를 'u'로 바꾸기
        text = text.replace('μ', 'u').replace('µ', 'u')
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

    def custom_split_texts(self, input_array):
        processed_array = []

        for item in input_array:
            # '_'로 분리된 요소들
            parts = item.split('_')

            # 각 부분에서 단일 문자열 제외하고 배열에 추가
            for part in parts:
                if len(part) > 1 or not part.isalpha():
                    processed_array.append(part)

        return processed_array

    def parse_pcb_bom_string_to_dict(self, bom_string):
        # 원래 문자열을 저장
        original_text = bom_string

        # 특수 문자를 공백으로 대체할 문자 정의
        special_chars = "`~!@$^&|\\=?:;'\",/<>,"

        # 숫자/숫자W 형태를 보존하기 위한 임시 대체 문자열
        temp_replacement = "TEMP_SLASH"

        # 숫자/숫자W 형태를 임시 문자열로 대체
        watt_pattern = re.compile(r'(\d+)/(\d+W)')
        bom_string = watt_pattern.sub(r'\1' + temp_replacement + r'\2', bom_string)

        # 괄호를 제외한 특수 문자를 공백으로 대체
        for char in special_chars:
            bom_string = bom_string.replace(char, " ")

        # 공백으로 문자열을 분할하여 단어 배열 생성
        parts = bom_string.split()

        # 임시 대체 문자열을 다시 슬래시로 변경
        parts = [part.replace(temp_replacement, '/') for part in parts]

        # 괄호 처리
        processed_parts = []
        for part in parts:
            if '(' in part and ')' in part:
                processed_parts.append(part.replace(f'({part[part.find("(") + 1:part.find(")")]})', ''))
                processed_parts.append(part[part.find("(") + 1:part.find(")")])
            else:
                processed_parts.append(part)

        # 데이터를 반환할 딕셔너리 생성
        return {
            "originText": original_text,
            "words": self.split_text(original_text)
        }

    def parse_pcb_bom_strings_to_dict(self, bom_strings):
        # 원래 문자열들을 저장 (공백으로 구분된 하나의 문자열로 결합)
        original_text = ' '.join(bom_strings)

        # 특수 문자를 공백으로 대체할 문자 정의
        special_chars = "`~!@$^&|\\=?:;'\",/<>,"

        # 숫자/숫자W 형태를 보존하기 위한 임시 대체 문자열
        temp_replacement = "TEMP_SLASH"

        processed_bom_strings = []
        for bom_string in bom_strings:
            # 숫자/숫자W 형태를 임시 문자열로 대체
            watt_pattern = re.compile(r'(\d+)/(\d+W)')
            bom_string = watt_pattern.sub(r'\1' + temp_replacement + r'\2', bom_string)

            # 괄호를 제외한 특수 문자를 공백으로 대체
            for char in special_chars:
                bom_string = bom_string.replace(char, " ")

            # 임시 대체 문자열을 다시 슬래시로 변경
            bom_string = bom_string.replace(temp_replacement, '/')

            processed_bom_strings.append(bom_string)

        # 공백으로 문자열을 분할하여 단어 배열 생성
        parts = []
        for string in processed_bom_strings:
            # parts.extend(string.split())
            # parts.extend(string)
            parts.append(string)

        # 괄호 처리
        processed_parts = []
        for part in parts:
            if part == '':
                continue
            if '(' in part and ')' in part:
                processed_parts.append(part.replace(f'({part[part.find("(") + 1:part.find(")")]})', ''))
                processed_parts.append(part[part.find("(") + 1:part.find(")")])
            else:
                processed_parts.append(part)

        # processed_parts 배열의 각 항목을 self.split_text로 처리하고 결과를 하나의 배열로 평탄화
        # flat_processed_parts = []
        # for part in processed_parts:
        #     flat_processed_parts.extend(self.split_text(part))

        # 데이터를 반환할 딕셔너리 생성
        return {
            "originText": original_text,
            "words": processed_parts
        }

    def classification(self, split_texts, reference_prefix=None):
        # split_text 이미 했다고 가정한다
        # 분할된 토큰을 기반으로 추가 분류 수행
        classification_result = self.parse_string(split_texts, reference_prefix)

        # 키 값을 제외하고 value만 포함하며, 'productName'의 값은 제외
        flat_classification = " ".join([value for key, value in classification_result.items() if key != 'productName'])

        # 분류 결과를 딕셔너리 형태로 변환하여 반환
        return {
            "originalClassification": classification_result,
            "flatClassification": flat_classification
        }

    def analyze_and_classify_bom(self, bom_item_list, reference_prefix=None):
        # Filtering and joining the query values
        query_list = [item['query'] for item in bom_item_list]
        # Parse the BOM string into its components
        parsed_data = self.parse_pcb_bom_strings_to_dict(query_list)

        # Use the parsed 'words' as input for further classification
        classification_result = self.classification(parsed_data['words'], reference_prefix)

        # Combine the results into a single dictionary
        return {
            "parsedData": parsed_data,
            "classificationResult": classification_result
        }

    def update_component_queries(self, data, excluded_targets):
        """
        Function to process data and update numeric queries.
        It counts entries where 'target' is 1 and 'query' matches "C/D/R/L + number".
        If the count differs from a numeric query (where 'target' is not 1, 4, 99, 100, or 98),
        it updates the numeric query by appending a unit based on the corresponding letter.
        """

        # Initialize counts for C, R, L
        target_1_counts = {'C': 0, 'R': 0, 'L': 0}

        # Iterate through the data to count C, R, L entries
        for entry in data:
            if entry['target'] == 1 and entry['query']:
                # Split the query values and count each type
                query_values = entry['query'].split(', ')
                for value in query_values:
                    if value[0] in ['C', 'R', 'L'] and all(c.isdigit() for c in value[1:]):
                        target_1_counts[value[0]] += 1

        update_query = ''
        # Update numeric queries
        for entry in data:
            if entry['target'] not in excluded_targets and entry['query'].isdigit():
                query_value = int(entry['query'])
                if query_value <= 500000:
                    for letter, count in target_1_counts.items():
                        if count and query_value != count:
                            suffix = {'R': 'Ohm', 'C': 'F', 'L': 'H'}.get(letter, '')
                            update_query = entry['query']
                            entry['query'] = entry['query'] + suffix

        return update_query

    def append_text_to_specific_format(self, input_string, append_text):
        regex = None
        text_to_append = ''

        if append_text == 'C':
            regex = re.compile(r"\b\d+[mkMunp]\b")
            text_to_append = 'F'
        elif append_text == 'R':
            # 숫자k 혹은 숫자m 형식과 숫자R 혹은 숫자r 형식을 찾습니다.
            regex = re.compile(r"\b\d+k\b|\b\d+m\b|\b\d+K\b|\b\d+M\b|\b\d+r\b|\b\d+R\b")
            text_to_append = 'Ohm'
        elif append_text == 'L':
            regex = re.compile(r"\b\d+m\b|\b\d+u\b|\b\d+M\b|\b\d+U\b")
            text_to_append = 'H'

        if regex is None:
            return None

        # 입력 문자열에서 모든 일치 항목을 찾아 해당 문자를 추가합니다.
        result = regex.sub(
            lambda match: match.group(0).rstrip("rR") + text_to_append if 'r' in match.group(0) or 'R' in match.group(
                0) else match.group(0) + text_to_append, input_string)

        # 만약 변경된 부분이 없으면 빈 문자열을 반환
        return result if result != input_string else ''

    def is_float(self, str):
        try:
            float(str)
            return True
        except ValueError:
            return False


    def is_part_number(self, s):
        """
        Further refined function to determine if a given string is a product name.
        It now accepts strings that are only numbers or a combination of numbers and non-unit characters.

        Criteria:
        1. Length of the string is at least 4.
        2. May contain '-' or parentheses.
        3. Contains a combination of letters and numbers.
        4. Not just a combination of (possibly floating) numbers and a single known unit.
        5. Accepts strings that are purely numbers or combinations of numbers and non-unit characters.
        """
        # Known units in PCB products
        units = ["Hz", "kHz", "MHz", "GHz", "V", "mV", "kV", "A", "mA", "μA",
                 "Ω", "kΩ", "MΩ", "W", "mW", "kW", "F", "μF", "nF", "pF", "H", "mH", "μH"]

        # Check if the string is just a combination of (possibly floating) numbers and a single known unit
        if any(re.fullmatch(rf'\d+(\.\d+)?\s*{unit}', s) for unit in units):
            return False

        # Accepts strings that are purely numeric or combinations of numbers and non-unit characters
        if re.fullmatch(r'\d+', s) or re.search(r'\d+[a-zA-Z]+', s):
            return True

        # The rest of the checks from the previous versions
        if len(s) < 4:
            return False

        if not re.search(r'[a-zA-Z]', s) and not re.search(r'[0-9]', s):
            return False

        if '-' in s or '(' in s or ')' in s:
            return True

        return True
