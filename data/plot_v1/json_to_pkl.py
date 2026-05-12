import json
import pickle

# JSON 파일 읽기
with open('data.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# 'text'와 'sentence' 필드를 처리하는 함수 정의
def process_item(item):
    if 'text' in item and isinstance(item['text'], str):
        # 문자열을 띄어쓰기로 분리하여 리스트로 변환
        item['text'] = item['text'].split()
    if 'sentence' in item and isinstance(item['sentence'], str):
        item['sentence'] = item['sentence'].split()
    return item

# 데이터 구조에 따라 처리
if isinstance(data, list):
    # 데이터가 리스트인 경우
    processed_data = [process_item(item) for item in data]
elif isinstance(data, dict):
    # 데이터가 딕셔너리인 경우
    processed_data = process_item(data)
else:
    raise TypeError("지원하지 않는 데이터 형식입니다.")

# Pickle 파일로 저장
with open('novel_train.pkl', 'wb') as pkl_file:
    pickle.dump(processed_data, pkl_file)