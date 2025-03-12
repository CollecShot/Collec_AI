import os
import csv
from PIL import Image
import pillow_heif
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
import os
from dotenv import load_dotenv

# .env 파일 로딩
load_dotenv()

def azure_authenticate():

    subscription_key = os.environ["VISION_KEY"]
    endpoint = os.environ["VISION_ENDPOINT"]
    computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))
    return computervision_client


def compress_image(input_file: str, max_dimension=1024, quality=80) -> str:
    file_name =  os.path.basename(input_file) 
    output_path = os.path.join("/Users/jiyoonjeon/projects/Collec_AI/dataset/AzureTestImgs/compressed/orig0", f"{os.path.splitext(file_name)[0]}_compressed.jpg")

    with Image.open(input_file) as img:
        if img.width > max_dimension or img.height > max_dimension:
            img.thumbnail((max_dimension, max_dimension))
        img.save(output_path, format="JPEG", quality=quality)

    return output_path
    


def classifify_tag(tags_str: str) -> str:
    """
    result가 아래 태그에 해당하면 해당 카테고리로 분류
    """
    #Shop
    shop_keywords = [
        "accessory", "bag", "clothing", "fashion", "fashion accessory",
        "luggage and bags", "design", "fashion design", "dress", 
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

    # keywords 리스트 중 하나라도 포함되면 True
    if any(keyword in tags_lower for keyword in shop_keywords):
        return "shop"
    elif any(keyword in tags_lower for keyword in place_keywords):
        return "place"
    elif any(keyword in tags_lower for keyword in animal_keywords):
        return "animal"
    elif any(keyword in tags_lower for keyword in people_keywords):
        return "people"
    else:
        return"other"


  


def get_tags_from_azure(input_folder, csv_path):
    """
    이미지 폴더 경로를 입력받아, Azure Computer Vision API로 태깅한 뒤
    결과를 CSV 파일로 저장하고, 결과 리스트를 반환한다.
    """
    client = azure_authenticate()

    results = []
    
    for input_file in os.listdir(input_folder):
        if not input_file.lower().endswith((".jpg", ".jpeg", ".png",".PNG", ".heic", ".heif", ".webp")):
            print(f"⚠️ 건너뜀: {input_file} (지원되지 않는 파일)")
            continue
    
        compressed_path = compress_image(os.path.join(input_folder, input_file))
        input_path=os.path.join(input_folder, input_file)

        with open(compressed_path, "rb") as local_image:
            tags_result = client.tag_image_in_stream(local_image)

        if len(tags_result.tags) == 0:
            tags_str = "No tags detected."
        else:
            tags_str = ", ".join(
                f"'{tag.name}' with confidence {tag.confidence*100:.2f}%"
                for tag in tags_result.tags
            )
            category=classifify_tag(tags_str)


        results.append([input_file, tags_str, category])


    # CSV로 저장
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["File Name", "Tags","Category"])
        writer.writerows(results)

    return results
