import os
from ui_detect import ui_detection

import os
from dotenv import load_dotenv
import argparse


load_dotenv()
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

def categorize_image(ui_classes):
    """
    UI detection 결과에 따라 카테고리를 지정
    """
    chat_keywords = {"chat_bubble"}
    music_keywords = {"forward button", "pause button", "play button", "rewind button", "shuffle"}
    coupon_keywords = {"barcode"}

    if any(keyword in ui_classes for keyword in chat_keywords):
        return "대화 & 메시지"
    if any(keyword in ui_classes for keyword in music_keywords):
        return "노래"
    if any(keyword in ui_classes for keyword in coupon_keywords):
        return "쿠폰 & 혜택"
    
    return "기타"


def roboflow_main(img_folder: str):


    # img_folder = "/Users/jiyoonjeon/projects/Collec_AI/dataset/RoboflowTestImgs"
    img_files = []

    for f in os.listdir(img_folder):
        file_lower = f.lower()
        if file_lower.endswith("png") or file_lower.endswith("jpg") or file_lower.endswith("jpeg"):
            img_files.append(f)

    results = {}
    for file_name in img_files:
        img_path = os.path.join(img_folder, file_name)

        ui_result = ui_detection(
                    image_path=img_path,
                    model_id="collec_250304/1",
                    api_url="https://detect.roboflow.com",
                    api_key=ROBOFLOW_API_KEY
                )
        
        ui_classes = {prediction["class"] for prediction in ui_result.get("predictions", [])}
        category = categorize_image(ui_classes)
        results[img_path]=category
        print(f"✅ {img_path} - UI Detect 결과: {ui_classes}, 분류된 카테고리: {category}")

        #dictionary형태의 result 반환. img_path: category
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Roboflow UI Detection")
    parser.add_argument("--img_folder", type=str, default=".", help="이미지 폴더 경로")
    args = parser.parse_args()

    roboflow_main(args.img_folder)
