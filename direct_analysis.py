# data_extractor.py

import numbers
import re
from urllib.parse import urlparse


class DirctAnalysis:
    def __init__(self, analyzer, extractor):
        self.analyzer = analyzer
        self.extractor = extractor

    @staticmethod  # 정적 메소드로 변경
    def calculate_none_percent(strings):
        # 주어진 문자열 목록에서 None 비율 계산
        none_count = sum(1 for item in strings if item is None or item == "None")
        none_percent = (none_count / len(strings)) * 100 if strings else 0
        return none_percent

    def percent_reference(self, item_list, percent, none_exclude=True):
        # 주어진 리스트에서 특정 패턴에 해당하는 비율 계산
        if not item_list:
            return 0

        pattern = re.compile(r"(?:^|,)[CRUJLSD]\d+(?:,|$)")
        none_percent = self.calculate_none_percent(item_list)

        if none_percent >= percent:
            return 0

        if none_exclude:
            item_list = [item for item in item_list if item is not None and item != "None"]

        matching = sum(bool(pattern.match(string)) for string in item_list)
        matching_percent = (matching / len(item_list)) * 100

        return matching_percent

    def percent_classification(self, item_list, none_exclude=True):
        # 주어진 리스트에서 분류 성공 비율 계산
        successful_classifications = 0

        if none_exclude:
            item_list = [item for item in item_list if item is not None and item != "None"]

        if not item_list:
            return 0

        for item_val in item_list:
            title_text = self.analyzer.split_text(item_val)
            classifications = self.analyzer.parse_string(title_text)
            key_count = len([key for key in classifications if key != 'productName'])

            if key_count >= 1:
                successful_classifications += 1

        success_percent = (successful_classifications / len(item_list)) * 100
        return success_percent

    def percent_package(self, str_list):
        # 주어진 문자열 리스트에서 패키지 성공 비율 계산
        if not str_list:
            return 0

        successful_package = 0
        for item_val in str_list:
            size = self.extractor.extract_size_from_title(item_val)
            if size:
                successful_package += 1

        success_percent = (successful_package / len(str_list)) * 100
        return success_percent

    def percent_none_for_is_pcb(self, item_list, percent):
        # PCB인 항목의 None 비율 계산
        none_count = 0
        filtered_list = [item for item in item_list if
                         not (isinstance(item, (int, float)) or (isinstance(item, str) and item.isdigit()))]

        for item_val in filtered_list:
            title_text = self.analyzer.split_text(item_val)
            classifications = self.analyzer.parse_string(title_text)
            key_count = len([key for key in classifications if key != 'productName'])

            if key_count >= 1:
                return False

            if item_val is None or item_val == "None":
                none_count += 1

        none_percent = (none_count / len(filtered_list)) * 100 if filtered_list else 0
        return none_percent >= percent

    @staticmethod  # 정적 메소드로 변경
    def percent_increment_digit_by_list(any_list):
        # 주어진 리스트가 숫자로만 이루어져 있고 증가하는지 확인
        is_digit = False
        pre_val = False
        increment_cnt = 0
        none_digit_cnt = 0
        any_list_len = len(any_list)
        for val in any_list:
            if val is None or val == 'None':
                any_list_len -= 1
                continue

            is_digit = False
            if isinstance(val, str) and val.isdigit():
                is_digit = True
                val = int(val)
            elif isinstance(val, numbers.Number):
                is_digit = True
                val = float(val)
            else:
                conversion_result, converted_val = DirctAnalysis.convert_to_int_if_no_decimal(val)
                if conversion_result:
                    is_digit = True
                    val = converted_val

            if not is_digit:
                continue

            if pre_val and pre_val < val:
                increment_cnt += 1

            pre_val = val

        if is_digit is False or increment_cnt == 0:
            return 0

        return (increment_cnt / (any_list_len - none_digit_cnt)) * 100

    @staticmethod
    def convert_to_int_if_no_decimal(val):
        try:
            converted_val = int(float(val))
            return True, converted_val
        except ValueError:
            return False, val

    @staticmethod  # 정적 메소드로 변경
    def percent_digit_by_list(any_list):
        # 주어진 리스트가 숫자로만 이루어져 있는지 확인
        any_list_len = len(any_list)
        digit_cnt = 0

        for val in any_list:
            if isinstance(val, str):
                try:
                    float(val)  # 문자열이 유효한 숫자인지 확인
                    digit_cnt += 1
                except ValueError:
                    pass  # 유효한 숫자가 아닌 경우 무시
            elif isinstance(val, numbers.Number):
                digit_cnt += 1
            elif val is None or val == 'None':
                any_list_len -= 1
                continue

        if any_list_len == 0:
            return 0
        return (digit_cnt / any_list_len) * 100

    @staticmethod  # 정적 메소드로 변경
    def starts_with_by_list(any_list, start_str):
        # 주어진 리스트의 요소가 특정 문자로 시작하는지 확인
        if not any_list:
            return 0

        start_count = sum(1 for val in any_list if isinstance(val, str) and val.startswith(start_str))
        return (start_count / len(any_list)) * 100

    @staticmethod  # 정적 메소드로 변경
    def is_valid_url(url):
        # URL의 유효성 검사
        parsed_url = urlparse(url)
        return bool(parsed_url.scheme) and bool(parsed_url.netloc)

    def percent_valid_urls(self, url_list):
        # 유효한 URL의 비율 계산
        if not url_list:
            return 0

        valid_urls = sum(self.is_valid_url(url) for url in url_list)
        valid_percent = (valid_urls / len(url_list)) * 100

        return valid_percent

    def find_reference_pattern(self, input_string):
        if input_string is None:
            return None

        # "C숫자", "R숫자", "L숫자" 형식을 찾기 위한 정규 표현식
        regex = r'[CRL]\d+'

        # 입력 문자열에서 모든 일치 항목을 찾습니다.
        matches = re.findall(regex, input_string)
        if matches:
            # 첫 번째 일치 항목의 첫 글자를 반환합니다.
            return matches[0][0]

        # 일치하는 패턴이 없으면 None을 반환합니다.
        return None
