import requests as reqs
import json


class SearchService:
    def find_update_part_number_average_score(self, results):
        # POST 요청을 보낼 URL
        url = 'http://localhost:8080/api/pcbItem/_searchList?target=2'

        for result in results:
            if result['predict'] == 2:
                item = result['predictResults']
                # POST 요청을 보내고 응답을 받음
                response = reqs.post(url, json=[l['query'] for l in item])

                if response.status_code == 200:
                    data = response.json().get('data', [])

                    # _score 값들을 추출
                    scores = [record.get('_score', 0) for record in data]

                    # _score 값이 없는 경우 건너뜀
                    if not scores:
                        continue

                    # 평균 점수 계산
                    average_score = sum(scores) / len(scores)

                    # 해당 항목에 averageScore 추가
                    result['averageScore'] = average_score

    def find_pcb_item_score(self, target, results):
        url = 'http://localhost:8080/api/pcbItem/_searchList?target=' + str(target)

        # POST 요청을 보내고 응답을 받음
        response = reqs.post(url, json=results)

        if response.status_code != 200:
            raise Exception(f"Request failed with status code {response.status_code}")

        data = response.json().get('data', [])

        # 결과를 저장할 사전
        result_scores = {}

        for record in data:
            query = record.get('query')
            if query in results:
                # _score 값들을 추출
                score = record.get('_score', 0)
                # 해당 query에 대한 결과에 추가
                result_scores[query] = score

        return result_scores

    def find_simple_pcb_item(self, results):
        url = 'http://localhost:8080/api/pcbItem/_searchList'

        # POST 요청을 보내고 응답을 받음
        response = reqs.post(url, json=results)

        if response.status_code != 200:
            raise Exception(f"Request failed with status code {response.status_code}")

        data = response.json().get('data', [])

        # 결과를 저장할 사전
        result_scores = {}

        records = []
        for record in data:
            query = record.get('query')
            if query in results:
                # _score 값들을 추출
                score = record.get('_score', 0)
                # 해당 query에 대한 결과에 추가
                result_scores[query] = score
                records.append(record)

        return records

    def find_pcb_item(self, bom_item_list):
        # POST 요청을 보낼 URL
        url = 'http://localhost:8080/api/pcbItem/_searchList'

        # query가 'None'이 아닌 항목들을 필터링
        valid_queries = [l for l in bom_item_list if
                         l['query'] != 'None' and type(l['query']) is not None and l['query'] != '']

        # valid_queries가 비어있는 경우 초기값 반환
        if not valid_queries:
            return {
                "target": None,
                "averageScore": 0.0,
                "isPcbItem": False
            }

        # 'None'으로 표시된 query의 개수-
        num_none_queries = len(bom_item_list) - len(valid_queries)

        # 서버에 POST 요청
        response = reqs.post(url, json=[l['query'] for l in valid_queries])
        if response.status_code == 200:
            # 응답된 JSON 데이터
            data = response.json().get('data', [])

            # target 값에 따른 빈도 계산
            target_freq = {}
            for item in data:
                target = item.get('target')
                if target in target_freq:
                    target_freq[target] += 1
                else:
                    target_freq[target] = 1

            # 가장 빈번한 target 값 찾기
            most_common_target = max(target_freq, key=target_freq.get)

            # 모든 객체의 _score 목록
            all_scores = [item['_score'] for item in data if '_score' in item]

            # 해당 target을 가진 객체들의 _score 평균 계산
            target_scores = [item['_score'] for item in data if
                             item.get('target') == most_common_target and '_score' in item]

            # 응답받지 못한 query 및 'None' query 개수 계산 (0점 처리)
            num_unanswered = len(valid_queries) - len(target_scores) + num_none_queries

            # 평균 점수 계산 (0점 포함)
            total_scores = target_scores + [0] * num_unanswered
            avg_score = sum(total_scores) / len(bom_item_list)

            # 결과를 딕셔너리로 반환
            return {
                "target": most_common_target,
                "averageScore": avg_score,
                "isPcbItem": any(score >= 85 for score in all_scores)
            }

    def check_is_pcb_item(self, bom_item_list):
        # POST 요청을 보낼 URL
        url = 'http://localhost:8080/api/pcbItem/_searchList'

        # query가 'None'이 아닌 항목들을 필터링
        valid_queries = [l for l in bom_item_list if
                         l['query'] != 'None' and type(l['query']) is not None and l['query'] != '']

        # valid_queries가 비어있는 경우 초기값 반환
        if not valid_queries:
            return {
                "averageScore": 0.0,
                "isPcbItem": False
            }

        # 'None'으로 표시된 query의 개수
        num_none_queries = len(bom_item_list) - len(valid_queries)

        # 서버에 POST 요청
        response = reqs.post(url, json=[l['query'] for l in valid_queries])
        if response.status_code == 200:
            # 응답된 JSON 데이터
            data = response.json().get('data', [])

            # 모든 객체의 _score 목록
            all_scores = [item['_score'] for item in data if '_score' in item]

            # 응답받지 못한 query 및 'None' query 개수 계산 (0점 처리)
            num_unanswered = len(valid_queries) - len(all_scores) + num_none_queries

            # 평균 점수 계산 (0점 포함)
            total_scores = all_scores + [0] * num_unanswered
            avg_score = sum(total_scores) / len(bom_item_list)

            # 결과를 딕셔너리로 반환
            return {
                "averageScore": avg_score,
                "isPcbItem": any(score >= 50 for score in all_scores)
            }

    def req_llama(self, prompt):
        # URL 및 헤더 설정
        url = 'http://localhost:11434/api/generate'
        headers = {'Content-Type': 'application/json'}
        # 데이터 구성
        data = {
            "model": "llama2specBom",
            "stream": False,
            "prompt": prompt
        }
        # POST 요청 보내기
        response = reqs.post(url, headers=headers, data=json.dumps(data))
        # 서버 응답을 JSON 형식으로 변환
        response_json = response.json()
        return response_json

    def search_octopart_mpn(self, query):
        url = "http://localhost:8099/api/searchLikePartsMpn"
        params = {'q': query}
        response = reqs.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            return data
        else:
            return f"Error: {response.status_code}"
