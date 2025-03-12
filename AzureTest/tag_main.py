import os
from tag import get_tags_from_azure

import os
from dotenv import load_dotenv

# .env 파일 로딩
load_dotenv()


def main():
    input_folder = "/Users/jiyoonjeon/projects/Collec_AI/dataset/AzureTestImgs/orig0"
    csv_path = "/Users/jiyoonjeon/projects/Collec_AI/results/results_csv/result3.csv"

    azure_results = get_tags_from_azure(input_folder, csv_path)
    
    print("✅ azure 분석 완료!!")

if __name__ == "__main__":
    main()