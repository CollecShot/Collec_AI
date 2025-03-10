import os
import csv
from PIL import Image
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

def compress_image(input_path: str, max_dimension=1024, quality=80) -> str:

    from os.path import dirname, basename, splitext, join

    filename = basename(input_path)
    output_path = join(dirname(input_path), f"{splitext(filename)[0]}_compressed.jpg")

    with Image.open(input_path) as img:
        # 축소
        if img.width > max_dimension or img.height > max_dimension:
            img.thumbnail((max_dimension, max_dimension))
        # JPEG로 저장 (압축)
        img.save(output_path, format="JPEG", quality=quality)

    return output_path

def get_tags_from_azure(img_paths, csv_path):
    """
    이미지 경로 목록을 입력받아, Azure Computer Vision API로 태깅한 뒤
    결과를 CSV 파일로 저장하고, 결과 리스트를 반환한다.
    """
    client = azure_authenticate()

    results = []
    for img_path in img_paths:
        filename = os.path.basename(img_path)
        compressed_path = compress_image(img_path)

        with open(compressed_path, "rb") as local_image:
            tags_result = client.tag_image_in_stream(local_image)

        if len(tags_result.tags) == 0:
            tags_str = "No tags detected."
        else:
            tags_str = ", ".join(
                f"'{tag.name}' with confidence {tag.confidence*100:.2f}%"
                for tag in tags_result.tags
            )

        results.append([filename, tags_str])

    # CSV로 저장
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["File Name", "Tags"])
        writer.writerows(results)

    return results
