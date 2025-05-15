import streamlit as st
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import resnet18
from ultralytics import YOLO
import matplotlib.pyplot as plt
from datetime import datetime
import os

# 定义保存结果的函数
def save_result(defects, frame, manual=False):
    """保存检测结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{'manual' if manual else 'auto'}_capture_{timestamp}.jpg"
    save_path = os.path.join("data", filename)  # 指定保存路径

    # 保存图片
    frame.save(save_path)

    # 记录检测结果
    if defects:
        main_defect = max(defects, key=lambda x: x['confidence'])
        record = {
            "time": timestamp,
            "file": filename,
            "defect_type": main_defect['type'],
            "confidence": main_defect['confidence']
        }
    else:
        record = {
            "time": timestamp,
            "file": filename,
            "defect_type": "正常",
            "confidence": 1.0
        }

    st.session_state.detection_history.append(record)
    st.toast(f"已保存检测结果：{filename}")

# ================== 模型加载部分 ==================
@st.cache_resource
def load_models():
    """加载预训练模型"""
    # YOLO模型
    yolo_model = YOLO(r'best.pt')  # 修改为实际路径

    # CNN模型
    cnn_model = resnet18(pretrained=False)
    cnn_model.fc = torch.nn.Linear(cnn_model.fc.in_features, 3)
    cnn_model.load_state_dict(torch.load('defect_cnn.pth', map_location='cpu'))
    cnn_model.eval()

    return yolo_model, cnn_model


yolo_model, cnn_model = load_models()

# ================== 核心检测函数 ==================
def cnn_classify(crop_img):
    """CNN分类处理"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(crop_img).unsqueeze(0)
    with torch.no_grad():
        output = cnn_model(img_tensor)
    return CLASS_NAMES[torch.argmax(output).item()]

def yolo_detect(frame, conf_threshold):
    """YOLO检测与结果处理"""
    results = yolo_model.predict(np.array(frame), conf=conf_threshold, verbose=False)[0]
    defects = []

    for box, cls, conf in zip(results.boxes.xyxy.cpu().numpy(),
                              results.boxes.cls.cpu().numpy().astype(int),
                              results.boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box)
        label = results.names[cls]

        # 记录检测结果
        defects.append({
            "type": label,
            "bbox": [x1, y1, x2, y2],
            "confidence": float(conf)
        })

        # 对logo区域进行CNN分类
        if label == 'logo':
            crop = frame.crop((x1, y1, x2, y2))
            cnn_result = cnn_classify(crop)
            if cnn_result != 'normal':
                defects.append({
                    "type": cnn_result,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": 0.8
                })

    return defects

# ================== 可视化函数 ==================
def draw_results(frame, defects):
    """绘制检测结果"""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(frame)

    for defect in defects:
        x1, y1, x2, y2 = defect['bbox']
        label = defect['type']

        # 绘制边界框
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # 绘制标签
        ax.text(x1, y1, label, color='white', fontsize=12, backgroundcolor='red')

    return fig

# ================== Streamlit界面修改部分 ==================
st.set_page_config(page_title="智能质检系统", page_icon="🔍", layout="wide")

# 初始化session状态
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []

# 侧边栏配置
with st.sidebar:
    st.header("系统配置")
    detection_mode = st.radio("检测模式", ["实时摄像头", "上传图片", "上传视频"])
    confidence_threshold = st.slider("置信度阈值", 0.0, 1.0, 0.5, 0.01)
    auto_save = st.checkbox("自动保存检测结果", False)
    manual_save = st.checkbox("启用空格键手动保存", True)

# 主界面
st.title("智能质检系统")
col1, col2 = st.columns(2)

with col1:
    st.subheader("原视频画面")
    original_camera_feed = st.empty()

with col2:
    st.subheader("检测后视频画面")
    detected_camera_feed = st.empty()

if detection_mode == "上传图片":
    uploaded_file = st.file_uploader("上传检测图片", type=["jpg", "png", "jpeg", "bmp"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        original_frame = image.copy()

        # 执行检测
        defects = yolo_detect(original_frame, confidence_threshold)
        detected_frame = draw_results(original_frame, defects)

        # 显示原视频和检测后视频
        col1.image(original_frame, caption="原图片", use_container_width=True)
        col2.pyplot(detected_frame)

        # 自动保存
        if auto_save:
            save_result(defects, detected_frame)

        # 手动保存处理
        if manual_save and st.button("保存当前图片"):
            save_result(defects, detected_frame, manual=True)

# 检测结果统计
with st.expander("检测结果统计"):
    if st.session_state.detection_history:
        latest = st.session_state.detection_history[-1]
        st.metric("最新缺陷类型", latest['defect_type'])
        st.metric("置信度", f"{latest['confidence'] * 100:.1f}%")
    else:
        st.warning("等待检测数据...")

    if st.session_state.detection_history:
        st.dataframe(pd.DataFrame(st.session_state.detection_history))
    else:
        st.info("暂无历史记录")

# 创建数据目录
os.makedirs("data", exist_ok=True)
