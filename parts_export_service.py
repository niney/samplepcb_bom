class PartsExportService(object):
    def to_camel_case(self, text):
        # 공백으로 분리된 각 단어의 첫 글자를 대문자로 변환
        return ''.join(word.capitalize() for word in text.split())

    def parse_key_value_by_llama(self, text):
        # 결과를 저장할 사전 생성
        result = {}

        # 각 줄을 순회하며 처리
        for line in text.split('\n'):
            # 콜론(:)을 기준으로 문자열을 분리
            parts = line.split(':')
            if len(parts) == 2:
                # 키에서 '*' 제거하고 앞뒤 공백을 없애기, 카멜 케이스로 변환
                key = self.to_camel_case(parts[0].replace('*', '').strip())
                # 값에서 앞뒤 공백을 없애기
                value = parts[1].strip()
                result[key] = value

        return result

    def extract_values_from_parsed_data(self, parsed_data, exclude_key_substrings=""):
        # 제외할 키 부분 문자열들을 포함하지 않는 value 값들만 추출하여 배열로 반환
        values = [value for key, value in parsed_data.items()
                  if value and not any(exclude_key.lower() in key.lower() for exclude_key in exclude_key_substrings)]

        return values

    def extract_values_by_analysis_result(self, data):
        # 우선 순위에 따라 체크할 필드 목록
        priority_fields = ['partNumber', 'candidatePartNumber', 'productName']
        combined_fields = ['watt', 'errorRange', 'ohm', 'parrot', 'voltage', 'temperature', 'size', 'henry', 'current',
                           'frequency']

        # priority_fields 필드를 먼저 확인
        for field in priority_fields:
            if field in data['originalClassification']:
                value = data['originalClassification'][field]
                if isinstance(value, list) and value:
                    return value[0]
                elif value:
                    return value

        # combined_fields 필드를 공백으로 구분하여 결합
        combined_values = []
        for field in combined_fields:
            if field in data['originalClassification']:
                value = data['originalClassification'][field]
                if value:
                    combined_values.append(str(value))

        # combined_values가 비어 있지 않다면 결합하여 반환
        if combined_values:
            return ' '.join(combined_values)

        # 모든 필드에 유효한 값이 없는 경우 빈 문자열 반환
        return ''
