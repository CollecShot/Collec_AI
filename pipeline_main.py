import os
import sys
import csv
import argparse

#module import
tag_main_path = os.path.join(".","AzureTest")
roboflow_path = os.path.join(".", "RoboflowTest")
ocr_path = os.path.join(".", "OCRTest")


if tag_main_path not in sys.path:
    sys.path.append(tag_main_path)

if roboflow_path not in sys.path:
    sys.path.append(roboflow_path)
    
if ocr_path not in sys.path:
    sys.path.append(ocr_path)

from tag_main import tag_main
from roboflow_main import roboflow_main
from OCR_main import OCR_main


def pipeline_main(img_path: str, output_csv: str):

    # --- A) Azure ---
    # get_tags_from_azure()는 [ [compressed_path, tags_str, category], ... ] 형태를 반환한다고 가정.
    # => 이때 compressed_path = "/어쩌구/.../XXX_compressed.jpg" 이런 식.
    # => 우리는 "원본 파일이름"과 매칭해야 편하므로, basename()을 써서 data_dict에 기록하자.
    compressed_img_path, azure_data = tag_main(
        img_path, 
        output_csv
        )

    data_dict = {}
    for (compressed_path, tags_str, azure_cat) in azure_data:
        # 예: compressed_path = "myImage_compressed.jpg"
        # basename = os.path.basename(compressed_path)


        data_dict[compressed_path] = {
            "azure_tags": tags_str,
            "azure_category": azure_cat,
            "roboflow_category": "",
            "ocr_text": "",
            "ocr_category": "",
            # 최종 category는 우선 Azure걸로 초기화
            "final_category": azure_cat
        }

        """
        카테고리가 1~4에 포함되면 끝내기

        """
        if azure_cat not in ["쇼핑&구매", "장소", "동물", "사람&인물"]:

        # roboflow
            rf_results = roboflow_main(compressed_img_path)  # dict
            for file_path, rf_cat in rf_results.items():
                # data_dict에 존재하면 업데이트, 없으면 새로 추가
                if file_path in data_dict:
                    data_dict[file_path]["roboflow_category"] = rf_cat
                    # "기타"가 아니면 최종카테고리로 확정정
                    if rf_cat != "기타":
                        data_dict[file_path]["final_category"] = rf_cat
                else:
                    data_dict[file_path] = {
                        "azure_tags": "",
                        "azure_category": "",
                        "roboflow_category": rf_cat,
                        "ocr_text": "",
                        "ocr_category": "",
                        "final_category": rf_cat
                    }
            if rf_cat == "기타": 

            # OCR
            # OCR_main()은 {파일이름: {"text": ..., "category": ...}, ...} 형태를 반환
                ocr_results = OCR_main(compressed_img_path)
                for file_path, ocr_info in ocr_results.items():
                    text = ocr_info["text"]
                    cat = ocr_info["category"]

                    if file_path in data_dict:
                        data_dict[file_path]["ocr_text"] = text
                        data_dict[file_path]["ocr_category"] = cat
                        if cat != "기타":
                            data_dict[file_path]["final_category"] = cat
                    else:
                        data_dict[file_path] = {
                            "azure_tags": "",
                            "azure_category": "",
                            "roboflow_category": "",
                            "ocr_text": text,
                            "ocr_category": cat,
                            "final_category": cat
                        }

    # 최종 csv 저장장
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "FilePath",
            "AzureTags",
            "AzureCategory",
            "RoboflowCategory",
            "OCRText",
            "OCRCategory",
            "FinalCategory"
        ])

        for file_path, info in data_dict.items():
            writer.writerow([
                file_path,
                info["azure_tags"],
                info["azure_category"],
                info["roboflow_category"],
                info["ocr_text"],
                info["ocr_category"],
                info["final_category"]
            ])

    print(f"✅ 파이프라인 완료! 결과 CSV: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="이미지 파이프라인 (Azure→Roboflow→OCR)")
    parser.add_argument("--img_path", type=str, required=True, help="이미지 폴더 경로")
    parser.add_argument("--output_csv", type=str, default="pipeline_result.csv", help="최종 CSV 파일명")
    args = parser.parse_args()

    pipeline_main(args.img_path, args.output_csv)

"""
python pipeline_main.py \
--img_folder "/Users/jiyoonjeon/projects/Collec_AI/dataset/AzureTestImgs/debug_classification" \
--output_csv "/Users/jiyoonjeon/projects/Collec_AI/results/results_csv/jiyoon_1.csv"

"""