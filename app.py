from AvengersEnsemble import *
from Draw import *
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av

# 페이지 기본 설정
st.set_page_config(
    # 페이지 제목
    page_title='MultiCampus Enjo2조',
    # 페이지 아이콘
    page_icon='app_gui/1.png'
)

# 공백
empty = st.empty()
empty.markdown('<div style="height: 200px;"></div>', unsafe_allow_html=True)

# 이미지와 제목을 한 줄에 나란히 표시하기 위해 column 두개로 나눔
col1, col2 = st.columns([2, 5])

# col1 위치에 이미지
with col1:
    st.image('app_gui/1.png', width=150)

# col2 위치에 프젝 이름
with col2:
    css_title = st.markdown("""
            <style>
                .title {
                    font-size: 70px;
                    font-weight: bold;
                    color: #f481512;
                    text-shadow: 3px  0px 0 #fff;}
            </style>
            <p class=title>
                AI 요리 비서 ✨
            </p>""", unsafe_allow_html=True)

# 공백
empty1 = st.empty()
empty1.markdown('<div style="height: 100px;"></div>', unsafe_allow_html=True)

def transform(frame: av.VideoFrame):
    # 프레임을 RGB로 변환
    img = frame.to_ndarray(format="bgr24")

    # 예측 수행
    boxes, confidences, labels = ensemble_predict(img)

    # 예측 결과를 프레임에 그리기
    draw(img, boxes, confidences, labels)
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# WebRTC 연결 설정: STUN/TURN 서버 설정
rtc_configuration = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}] # STUN 서버
    # TURN 서버: 선택사항
}

# 스트리밍 ui
webrtc_streamer(key="streamer", video_frame_callback=transform, rtc_configuration=rtc_configuration, sendback_audio=False)