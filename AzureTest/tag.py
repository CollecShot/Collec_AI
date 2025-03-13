import os
import csv
from PIL import Image
import pillow_heif
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
import os
from dotenv import load_dotenv
import datetime 
import re

# .env 파일 로딩
load_dotenv()

#날짜 기준으로 폴더 생성(코드를 하루에 한 번만 돌린다고 가정)
today_str = datetime.datetime.now().strftime("%Y%m%d")


compressed_folder = f"./dataset/compressed/compressed_{today_str}/"

os.makedirs(compressed_folder, exist_ok=True)

def azure_authenticate():

    subscription_key = os.environ["VISION_KEY"]
    endpoint = os.environ["VISION_ENDPOINT"]
    computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))
    return computervision_client


def compress_image(input_file: str, max_dimension=1024, quality=80) -> str:
    file_name =  os.path.basename(input_file) 

    """
    compress folder를 지정해줘야되나? 
    파이프라인을 돌릴 때, 시간 순으로 compress_folder를 만들어. 
    azure, roboflow, ocr이 이 폴더 내 imgs를 기반으로 돌아가.. 
    그럼?
    자동화 파이프라인을 어케 만들지? 
    백에서 요청을 할 때 파이프라인이 동작. 
    폴더를 요청한다고 치면 -> 이게 input folder고,
    이 input_folder를 전처리해서 output folder에 들어가는건데, 그럼 경로 설정을 어케하지.
    음 . . . result폴더에 시간 기준으로 폴더명 만들기? 

    1. BE-> AI에 요청 시, 폴더를 요청한다고 쳐.
    2. result 폴더에 전처리 폴더 생성 ex)"compress_250313" 이런 식으로. 
       -> 이걸 폴더에 받아서 써. 그럼 이 함수에서는 compress폴더 경로만 추가하면 됨.


    """

    output_path = os.path.join(compressed_folder, f"{os.path.splitext(file_name)[0]}_compressed.jpg")
    
    # output_path = os.path.join("/Users/jiyoonjeon/projects/Collec_AI/dataset/AzureTestImgs/compressed/orig0", f"{os.path.splitext(file_name)[0]}_compressed.jpg")

    with Image.open(input_file) as img:
        if img.width > max_dimension or img.height > max_dimension:
            img.thumbnail((max_dimension, max_dimension))
        img.save(output_path, format="JPEG", quality=quality)

    return output_path
    

def contains_keyword(tags_lower, keywords):
    for keyword in keywords:
        # 정규 표현식: 단어 경계(\b)를 활용하여 정확한 단어 매칭
        if re.search(r'\b' + re.escape(keyword) + r'\b', tags_lower):
            print(f"🔍 Found match: {keyword}")
            return True
    return False


def classifify_tag(tags_str: str) -> str:
    """
    result가 아래 태그에 해당하면 해당 카테고리로 분류
    """
    shop_keywords = [
        "accessory", "bag", "clothing", "fashion", "fashion accessory",
        "luggage and bags", "fashion design", "dress", 
        "cosmetics", "footwear", "furniture", "online advertising"
    ]
    place_keywords= [
        "sky", "outdoor", "cloud", "building", "lighthouse", "night", 
        "landmark", "city", "road", "mountain", "ground", "tree", "water", 
        "beach", "plant", "nature", "sunset", "crosswalk", "way", 
        "architecture", "street", "vehicle"
    ]
    animal_keywords= [
        "indoor", "animal", "pet", "mammal", "dog", "cat", "hamster",
        "rodent", "rat", "bird", "small to medium-sized cats",
        "outdoor", "pigeon", "ground", "feather", "amphibian",
        "reptile", "terrier", "whiskers"
    ]
    people_keywords= [
        "human face", "person", "woman", "man", "smile", "footwear",
        "girl", "boy", "group", "collage", "lip", "tooth", "eyelash",
        "wall", "hat"
    ]

    tags_lower = tags_str.lower()
    print(f"➡️{tags_lower}")

    # # keywords 리스트 중 하나라도 포함되면 True
    # if any(keyword in tags_lower for keyword in shop_keywords):
    #     return "shop"
    # elif any(keyword in tags_lower for keyword in place_keywords):
    #     return "place"
    # elif any(keyword in tags_lower for keyword in animal_keywords):
    #     return "animal"
    # elif any(keyword in tags_lower for keyword in people_keywords):
    #     return "people"
    # else:
    #     return "other"

    
    if contains_keyword(tags_lower, shop_keywords):
        return "쇼핑&구매"
    elif contains_keyword(tags_lower, place_keywords):
        return "장소"
    elif contains_keyword(tags_lower, animal_keywords):
        return "동물"
    elif contains_keyword(tags_lower, people_keywords):
        return "사람&인물"
    else:
        return "기타"

    
    


def get_tags_from_azure(input_img, csv_path):
    """
    이미지 폴더 경로를 입력받아, Azure Computer Vision API로 태깅한 뒤
    결과를 CSV 파일로 저장하고, 결과 리스트를 반환한다.
    """
    client = azure_authenticate()

    results = []
    
    if not input_img.lower().endswith((".jpg", ".jpeg", ".png",".PNG", ".heic", ".heif", ".webp")):
        print(f"⚠️ 건너뜀: {input_img} (지원되지 않는 파일)")
    
    compressed_path = compress_image(input_img)

    with open(compressed_path, "rb") as local_image:
        tags_result = client.tag_image_in_stream(local_image)
            

    if len(tags_result.tags) == 0:
        tags_str = "No tags detected."
    else:
        tags_str = ", ".join(
            f"'{tag.name}' with confidence {tag.confidence*100:.2f}%"
            for tag in tags_result.tags
        )
        print(f"📍{compressed_path} : tags_result",tags_str)
        category=classifify_tag(tags_str)
        print(f"💡{category}")


    results.append([compressed_path, tags_str, category])


    # CSV로 저장
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["File dir", "Tags","Category"])
        writer.writerows(results)

    return compressed_path, results
