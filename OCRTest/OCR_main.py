import os
import cv2
import re
import numpy as np
from google.cloud import vision
from PIL import Image, ImageDraw


os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/Users/jiyoonjeon/projects/Collec_AI/wise-alpha-451006-r6-ffa3465ce044.json"

def detect_text(input_file, input_folder):
    """
    Detects text in the file
    """
    client = vision.ImageAnnotatorClient()

    path = os.path.join(input_folder, input_file)
    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print("Texts:")

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )
    return texts

def clean_text(text):
    """
    Removes unnecessary newlines and spaces from OCR text
    """
    #1. 연속된 개행(\n) 및 공백을 단일 공백으로 변환
    text = re.sub(r'\s+', ' ', text)

    # 2. 특정 패턴(한글과 숫자, 한글과 영어) 사이에는 공백 추가 (너무 붙지 않도록)
    text = re.sub(r'([가-힣])(\d)', r'\1 \2', text)  # 한글 + 숫자
    text = re.sub(r'(\d)([가-힣])', r'\1 \2', text)  # 숫자 + 한글
    text = re.sub(r'([a-zA-Z])([가-힣])', r'\1 \2', text)  # 영문 + 한글
    text = re.sub(r'([가-힣])([a-zA-Z])', r'\1 \2', text)  # 한글 + 영문

    return text.strip()  # 앞뒤 공백 제거



def extract_text(text_info):
    """
    Exstracts text from Google Vision's API response
    """
    descriptions = []
    for text in text_info:
        description = text.description
        descriptions.append(description)

    raw_text = ' '.join(descriptions)  # 기본적으로 한 줄로 합치기
    return clean_text(raw_text)  # 클린 텍스트로 변환


def classify_text(text, threshold = 800):
    """
    Classify img based on text length and keywords
    """
    booking_keywords = ['예약', "예매", "티켓", "거래", "주문", "내역", "신용", "체크"]
    if len(text) >= threshold :
        return "문서 & 정보"

    if any(keyword in text for keyword in booking_keywords):
        return "예약 & 거래"
    
    return "기타"

    
def main():
    input_folder = "/Users/jiyoonjeon/projects/Collec_AI/dataset/OCRTestImgs"
    output_folder = "/Users/jiyoonjeon/projects/Collec_AI/results/results_ocr"

    results = {}


    for input_file in os.listdir(input_folder):
        text_info = detect_text(input_file, input_folder)
        extracted_text = extract_text(text_info)
        print(f"🚨{input_file}->{len(extracted_text)}")
        category = classify_text(extracted_text)
        # save result in dict
        results[input_file] ={
            "text" : extracted_text,
            "category" : category
        }

        # save result in txt file
        output_file = os.path.join(output_folder,f"{os.path.splitext(input_file)[0]}.txt")
        with open (output_file, "w", encoding = "utf-8") as f:
                f.write(f"{category}\n\n{extracted_text}")
        print(f"✅ {input_file} - 텍스트 감지 완료")
         
    return results


if __name__ == "__main__":
    main()
    


