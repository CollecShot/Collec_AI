from inference_sdk import InferenceHTTPClient

def ui_detection(image_path: str, model_id: str, api_url: str, api_key: str):
    """
    UI detection API를 사용해서 결과를 받는 예시 함수.
    """
    CLIENT = InferenceHTTPClient(
        api_url=api_url,
        api_key=api_key
    )
    # 실제 inference
    result = CLIENT.infer(image_path, model_id = model_id)
    return result
