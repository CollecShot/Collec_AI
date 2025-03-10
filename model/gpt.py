from openai import OpenAI
import os
from dotenv import load_dotenv

# .env 파일 로딩
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key = OPENAI_API_KEY
)

# Define the system role and model
model = 'gpt-4o-mini'


# system_role 설정
system_role = (
    "당신은 데이터를 카테고리로 분류하는 일을 수행합니다.\n"
    "입력으로 들어오는 데이터는 이미지 detection을 한 결과물입니다.\n"
    "당신은 데이터를 기준으로 아래 카테고리 중 하나에 매핑합니다.\n\n"
    "카테고리:\n"
    "1. 쇼핑&구매\n"
    "2. 문서&정보\n"
    "3. 거래&예약\n"
    "4. 장소\n"
    "5. 쿠폰&혜택\n"
    "6. 대화&메시지\n"
    "7. 노래\n"
    "8. 동물\n"
    "9. 사람&인물\n\n"
    "답변은 반드시 위 9가지 카테고리 중 하나로만 골라서 말해 주세요."
)

def classify_image_detection(detection_result: str) -> str:
    """
    detection_result(예: 'food' with confidence 99.80%, 'meal' with confidence 99.21%, ...)을
    OpenAI API로 분류해, 9가지 중 하나의 카테고리로 반환한다.
    """
    user_prompt = f"다음 이미지 detection 결과를 바탕으로, 가장 적절한 카테고리를 골라주세요:\n\n{detection_result}"

    # ChatCompletion API 호출
    response = client.chat.completion.create(
        model=model,  
        messages=[
            {"role": "system", "content": system_role},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0  # 분류 태스크이므로 답변을 일정하게 유지하기 위해 0 또는 낮은 값 사용
    )

    # 응답 메시지
    classification = response["choices"][0]["message"]["content"].strip()
    return classification

