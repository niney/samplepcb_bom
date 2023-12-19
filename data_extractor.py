# data_extractor.py

import json
import re


class DataExtractor:

    def __init__(self):
        with open('partsPackageOnly.json', 'r', encoding='utf-8') as f:
            self.sizes = [size for size in json.load(f) if len(size) > 1]
        self._split_sizes()  # 초기화할 때 한 번만 실행됩니다.

    def _split_sizes(self):
        # sizes를 숫자 형식과 그 외로 분리하는 메서드
        num_pattern = re.compile(r'\d+(\.\d+)?')  # 숫자 형식만 찾는 정규 표현식
        self.numeric_sizes = [size for size in self.sizes if num_pattern.match(size)]
        self.other_sizes = [size for size in self.sizes if size not in self.numeric_sizes]

    def extract_size_from_title(self, titles):
        if isinstance(titles, list):
            title_words = [word.lower() for word in titles]
        elif isinstance(titles, str):
            title_words = titles.lower().split()
        else:
            raise ValueError("titles should be either a string or a list")

        # 숫자 형식의 사이즈를 우선적으로 찾기
        for size in self.numeric_sizes:
            for word in title_words:
                if size in word:
                    return size

        # 그 외의 사이즈 포맷 찾기
        for size in self.other_sizes:
            if size.lower() in title_words:
                return size

        return None
