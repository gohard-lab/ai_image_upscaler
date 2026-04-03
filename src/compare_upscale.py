import cv2
import time
from tracker_exe import log_app_usage

# -------------------------------------------------------------------
# [1] 단순 확대 (전통적인 보간법)
# 대본: "주변 색을 대충 섞기 때문에 오히려 더 뿌옇게 보입니다."
# -------------------------------------------------------------------
def resize_basic(image_path, scale=4):
    img = cv2.imread(image_path)
    
    # 픽셀을 강제로 늘리고 사이사이를 단순히 계산해서 채워 넣음 (Bicubic)
    result = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return result


# -------------------------------------------------------------------
# [2] 교과서적 정답 (PyTorch 딥러닝)
# 대본: "일반 사무용 노트북에서는 GPU 메모리 부족으로 뻗어버립니다."
# -------------------------------------------------------------------
def resize_pytorch_textbook(image_path):
    """
    이론상 훌륭하지만, 일반 노트북에서는 
    RuntimeError: CUDA out of memory 에러를 뿜으며 프로그램이 강제 종료되는 방식입니다.
    """
    import torch
    # 무거운 모델을 그래픽카드(VRAM)에 통째로 올리려다 메모리 초과 발생
    # model = HeavySRModel().cuda() 
    # tensor_img = load_image_to_tensor(image_path).cuda()
    # return model(tensor_img)
    pass


# -------------------------------------------------------------------
# [3] 현업의 타협점 (OpenCV DNN) 
# 대본: "용량은 수십 메가에 불과하지만, CPU만으로도 기가 막힌 결과를 뽑아냅니다."
# -------------------------------------------------------------------
def resize_opencv_dnn(image_path, model_path="FSRCNN_x4.pb", scale=4):
    img = cv2.imread(image_path)
    
    # OpenCV 내장 DNN 모듈을 활용한 경량화 AI 추론
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("fsrcnn", scale)
    
    # CPU 연산만으로 픽셀 사이의 디테일을 AI가 직접 그려 넣음
    result = sr.upsample(img)
    
    # [데이터 트래커] 사용 현황 기록 (필수)
    log_app_usage("ai_upscaler_exe", "upscale_completed", details={
        "method": "OpenCV_DNN_FSRCNN",
        "scale_factor": scale
    })
    
    return result

# =====================================================================
# 🌟 [GitHub Star 부탁드립니다!]
# 한글: 이 코드가 여러분의 퇴근 시간을 조금이라도 단축시켜 드렸다면, 조용히 소스코드만 가져가시는 체리피커가 되기보단 제 깃허브 저장소에 Star(⭐)를 꾹 눌러주세요! 여러분의 클릭 한 번이 더 나은 오픈소스 생태계를 만듭니다.
# English: If this project saved your time, please consider leaving a Star(⭐) on my GitHub repository. Don't be a silent cherry-picker—your support drives open-source developers to create better tools!
# =====================================================================