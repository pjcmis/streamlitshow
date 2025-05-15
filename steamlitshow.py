import streamlit as st
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import time
from datetime import datetime
import os
import torch
from torchvision import transforms
from torchvision.models import resnet18
from ultralytics import YOLO
from streamlit.components.v1 import html


import importlib
importlib.import_module('cv2')

# 定义保存结果的函数
def save_result(defects, frame, manual=False):
    """保存检测结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{'manual' if manual else 'auto'}_capture_{timestamp}.jpg"
    save_path = os.path.join("data", filename)  # 指定保存路径

    # 保存图片
    cv2.imwrite(save_path, frame)

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
    yolo_model = YOLO(
        r'best.pt'
    )  # 修改为实际路径

    # CNN模型
    cnn_model = resnet18(pretrained=False)
    cnn_model.fc = torch.nn.Linear(cnn_model.fc.in_features, 3)
    cnn_model.load_state_dict(torch.load('defect_cnn.pth', map_location='cpu'))
    cnn_model.eval()

    return yolo_model, cnn_model


yolo_model, cnn_model = load_models()

# ================== 检测参数 ==================
CLASS_NAMES = ['dong', 'que', 'normal']
STYLE_CONFIG = {
    'logo': {
        'thickness': 4,
        'font_scale': 2.4,
        'font_thickness': 5
    },
    'default': {
        'thickness': 2,
        'font_scale': 0.8,
        'font_thickness': 2
    }
}
COLORS = {
    'logo': (255, 0, 0),  # 红色
    'mao': (255, 165, 0),  # 橙色
    'dong': (0, 0, 255),  # 蓝色
    'que': (0, 255, 0)  # 绿色
}


# ================== 核心检测函数 ==================
def cnn_classify(crop_img):
    """CNN分类处理"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(Image.fromarray(crop_img)).unsqueeze(0)
    with torch.no_grad():
        output = cnn_model(img_tensor)
    return CLASS_NAMES[torch.argmax(output).item()]


def yolo_detect(frame, conf_threshold):
    """YOLO检测与结果处理"""
    results = yolo_model.predict(frame, conf=conf_threshold, verbose=False)[0]
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
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                cnn_result = cnn_classify(cv2.cvtColor(crop,
                                                       cv2.COLOR_BGR2RGB))
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
    for defect in defects:
        x1, y1, x2, y2 = defect['bbox']
        label = defect['type']
        style = STYLE_CONFIG.get(label, STYLE_CONFIG['default'])

        # 绘制边界框
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[label],
                      style['thickness'])

        # 绘制标签
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                      style['font_scale'],
                                      style['font_thickness'])
        # 检查文字位置是否在视频画面内
        if y1 - th - 5 < 0:
            y1 = th + 5
        if x1 + tw > frame.shape[1]:
            x1 = frame.shape[1] - tw

        cv2.rectangle(frame, (x1, y1 - th - 5), (x1 + tw, y1), COLORS[label],
                      -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    style['font_scale'], (255, 255, 255),
                    style['font_thickness'])
    return frame


# ================== Streamlit界面修改部分 ==================
def add_space_key_listener():
    """添加空格键监听"""
    space_key_js = """
    <script>
    document.addEventListener('keydown', function(event) {
        if (event.code === 'Space') {
            window.parent.postMessage({'type':'streamlit:setComponentValue', 'value':'space_pressed'}, '*');
        }
    });
    </script>
    """
    html(space_key_js, height=0, width=0)


# 初始化页面
st.set_page_config(page_title="智能质检系统", page_icon="🔍", layout="wide")
add_space_key_listener()

# 初始化session状态
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'capture' not in st.session_state:
    st.session_state.capture = False
if 'video_playing' not in st.session_state:
    st.session_state.video_playing = False
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = 0

# 侧边栏配置
with st.sidebar:
    st.header("系统配置")
    detection_mode = st.radio("检测模式", ["实时摄像头", "上传图片", "上传视频"])  # 添加上传视频选项
    confidence_threshold = st.slider("置信度阈值", 0.0, 1.0, 0.5, 0.01)
    auto_save = st.checkbox("自动保存检测结果", False)  # 默认关闭自动保存
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

if detection_mode == "实时摄像头":
    start_btn = st.button("开始检测")
    stop_btn = st.button("停止检测")
    save_btn = st.button("保存当前图片")  # 添加保存按钮

    if start_btn:
        st.session_state.capture = True
        cap = cv2.VideoCapture(1)  # 根据摄像头调整编号

    if stop_btn:
        st.session_state.capture = False
        if 'cap' in locals():
            cap.release()

    if st.session_state.capture:
        while st.session_state.capture:
            ret, frame = cap.read()
            if not ret:
                st.error("摄像头连接失败")
                break

            original_frame = frame.copy()  # 保存原视频帧

            # 执行检测
            defects = yolo_detect(frame, confidence_threshold)
            detected_frame = draw_results(frame.copy(), defects)

            # 显示原视频和检测后视频
            original_camera_feed.image(cv2.cvtColor(original_frame,
                                                    cv2.COLOR_BGR2RGB),
                                       channels="RGB",
                                       use_container_width=True)
            detected_camera_feed.image(cv2.cvtColor(detected_frame,
                                                    cv2.COLOR_BGR2RGB),
                                       channels="RGB",
                                       use_container_width=True)

            # 保存逻辑
            if auto_save:
                save_result(defects, detected_frame)

            # 手动保存处理
            if manual_save and (st.session_state.get('space_pressed')
                                or save_btn):
                save_result(defects, detected_frame, manual=True)
                st.session_state.space_pressed = False

elif detection_mode == "上传图片":
    uploaded_file = st.file_uploader("上传检测图片",
                                     type=["jpg", "png", "jpeg", "bmp"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        frame = np.array(image.convert('RGB'))

        original_frame = frame.copy()  # 保存原视频帧

        # 执行检测
        defects = yolo_detect(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                              confidence_threshold)
        detected_frame = draw_results(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                                      defects)

        # 显示原视频和检测后视频
        col1.image(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB),
                   caption="原图片",
                   use_container_width=True)
        col2.image(cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB),
                   caption="检测后图片",
                   use_container_width=True)

        # 自动保存
        if auto_save:
            save_result(defects, detected_frame)

        # 手动保存处理
        if manual_save and st.button("保存当前图片"):
            save_result(defects, detected_frame, manual=True)

elif detection_mode == "上传视频":  # 处理上传视频的逻辑
    uploaded_video = st.file_uploader("上传检测视频", type=["mp4", "avi"])
    if uploaded_video is not None:
        # 保存上传的视频到临时文件
        with open('temp_video.mp4', 'wb') as f:
            f.write(uploaded_video.read())

        cap = cv2.VideoCapture('temp_video.mp4')
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 播放/暂停按钮
        if st.session_state.video_playing:
            play_pause_button = st.button("暂停")
        else:
            play_pause_button = st.button("播放")

        if play_pause_button:
            st.session_state.video_playing = not st.session_state.video_playing

        # 添加进度条
        st.session_state.current_frame = st.slider(
            "视频进度", 0, total_frames - 1, st.session_state.current_frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame)

        while True:
            if st.session_state.video_playing:
                ret, frame = cap.read()
                if ret:
                    st.session_state.current_frame = int(
                        cap.get(cv2.CAP_PROP_POS_FRAMES))
                    original_frame = frame.copy()  # 保存原视频帧

                    # 执行检测
                    defects = yolo_detect(frame, confidence_threshold)
                    detected_frame = draw_results(frame.copy(), defects)

                    # 显示原视频和检测后视频
                    original_camera_feed.image(cv2.cvtColor(
                        original_frame, cv2.COLOR_BGR2RGB),
                                               channels="RGB",
                                               use_container_width=True)
                    detected_camera_feed.image(cv2.cvtColor(
                        detected_frame, cv2.COLOR_BGR2RGB),
                                               channels="RGB",
                                               use_container_width=True)

                    # 保存逻辑
                    if auto_save:
                        save_result(defects, detected_frame)

                    # 手动保存处理
                    if manual_save and (st.session_state.get('space_pressed')):
                        save_result(defects, detected_frame, manual=True)
                        st.session_state.space_pressed = False
                else:
                    st.session_state.video_playing = False
                    break
            else:
                time.sleep(0.1)

        cap.release()
        os.remove('temp_video.mp4')  # 删除临时文件

# 检测结果统计
with st.expander("检测结果统计"):
    # 实时统计
    if st.session_state.detection_history:
        latest = st.session_state.detection_history[-1]
        st.metric("最新缺陷类型", latest['defect_type'])
        st.metric("置信度", f"{latest['confidence'] * 100:.1f}%")
    else:
        st.warning("等待检测数据...")

    # 历史记录
    if st.session_state.detection_history:
        st.dataframe(pd.DataFrame(st.session_state.detection_history))
    else:
        st.info("暂无历史记录")

# 创建数据目录
os.makedirs("data", exist_ok=True)
