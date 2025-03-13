import os
from tag import get_tags_from_azure

import os
from dotenv import load_dotenv
import argparse

# .env 파일 로딩
load_dotenv()

"""
코드 돌릴 때 이런 식으로 쓰기!
python my_script.py --img_folder /Users/jiyoonjeon/projects/Collec_AI/dataset/RoboflowTestImgs
"""


def tag_main(input_img: str, output_csv: str):


    print(f"input 이미지 폴더 경로는 {input_img}, output csv 경로는 {output_csv} 입니다.")
    azure_results = get_tags_from_azure(input_img, output_csv)
    

    print("✅ azure 분석 완료!!")
    return azure_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Azure Tagging Script")
    parser.add_argument("--input_img", type=str, required=True, help="이미지 폴더 경로")
    parser.add_argument("--output_csv", type=str, default="./results/results_csv/temp_azure.csv", help="결과 CSV 파일명")
    args = parser.parse_args()
    
    tag_main(args.input_img, args.output_csv)