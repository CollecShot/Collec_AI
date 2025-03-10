import os
from azure import get_tags_from_azure
from gpt import classify_image_detection
from ui_detect import ui_detection

import os
from dotenv import load_dotenv

# .env 파일 로딩
load_dotenv()

# 이제 os.getenv("키이름") 형태로 값을 가져올 수 있어
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

def main():
    # 1) 분석할 이미지 리스트
    img_paths = [
        "C:\\Users\\mjw35\\Documents\\2025\\Collec\\AI test dataset\\이미지 분석\\5.jpg"
    ]
    csv_path = "C:/Users/mjw35/Documents/2025/Collec/result.csv"

    # 2) Azure API 사용 → 태깅 → CSV 저장 + 결과 반환
    azure_results = get_tags_from_azure(img_paths, csv_path)
    # azure_results = [['5.jpg', "'food' with confidence 99.80%, 'meal' with confidence 99.21%, ..."]]

    # 3) OpenAI API 사용 → 카테고리 분류
    #    여러 장의 이미지를 처리한다면, 각각 태그 문자열에 대해 분류 로직을 돌리면 된다.
    for file_name, tags_str in azure_results:
        category = classify_image_detection(tags_str)
        print(f"파일 {file_name} 분류 결과: {category}")

        # 4) 카테고리에 따라 분기 처리
        #   - (3-1) 쇼핑&구매, 장소, 동물, 사람&인물 → 최종 결과
        #   - (3-2) 쿠폰&혜택, 대화&메시지, 노래 → UI detect API
        if category in ["쇼핑&구매", "장소", "동물", "사람&인물"]:
            # 최종 결과 처리(프린트만 해줄게)
            print(f"이미지 {file_name}은 최종 카테고리 '{category}'로 결정되었습니다.\n")
        elif category in ["쿠폰&혜택", "대화&메시지", "노래"]:
            print(f"'{category}' 카테고리로 분류. UI Detect API 실행 중...")
            ui_result = ui_detection(
                image_path=file_name,
                model_id="collec_250304/1",
                api_url="https://detect.roboflow.com",
                api_key=ROBOFLOW_API_KEY
            )
            print("UI Detect API 결과:", ui_result)
        else:
            # 3. 거래&예약, 2. 문서&정보 등등
            print(f"'{category}' 카테고리는 별도 처리 로직이 필요합니다.")

if __name__ == "__main__":
    main()
