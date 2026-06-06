import sys
import os
import streamlit as st

# Thêm thư mục src vào path để import predict.py
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from predict import MBTIPredictor
MBTI_DESC = {
    'INTJ': 'Nhà kiến trúc — chiến lược, độc lập, lý trí',
    'INTP': 'Nhà tư duy — logic, tò mò, sáng tạo',
    'ENTJ': 'Nhà chỉ huy — quyết đoán, lãnh đạo',
    'ENTP': 'Người tranh luận — thông minh, năng động',
    'INFJ': 'Người cố vấn — sâu sắc, lý tưởng',
    'INFP': 'Người hòa giải — giàu cảm xúc, sáng tạo',
    'ENFJ': 'Người chỉ đường — ấm áp, truyền cảm hứng',
    'ENFP': 'Người truyền cảm hứng — nhiệt huyết, tự do',
    'ISTJ': 'Người trách nhiệm — thực tế, đáng tin cậy',
    'ISFJ': 'Người bảo vệ — tận tâm, chu đáo',
    'ESTJ': 'Nhà điều hành — tổ chức, kỷ luật',
    'ESFJ': 'Người quan tâm — hòa đồng, tận tụy',
    'ISTP': 'Nhà kỹ thuật — linh hoạt, thực tế',
    'ISFP': 'Người nghệ sĩ — nhẹ nhàng, yêu cái đẹp',
    'ESTP': 'Người năng động — táo bạo, thực tế',
    'ESFP': 'Người trình diễn — vui vẻ, tự phát',
}
QUESTIONS = [
    {
        'key': 'q1',
        'label': '1. Describe your ideal weekend or day off.',
        'placeholder': 'e.g. I love staying home, reading books and thinking quietly alone...'
    },
    {
        'key': 'q2',
        'label': '2. How do you usually make important decisions?',
        'placeholder': 'e.g. I analyze all possibilities logically before deciding...'
    },
    {
        'key': 'q3',
        'label': '3. How do you feel about social gatherings and meeting new people?',
        'placeholder': 'e.g. I enjoy small groups but large parties drain my energy...'
    },
    {
        'key': 'q4',
        'label': '4. How do you handle stress or difficult situations?',
        'placeholder': 'e.g. I tend to withdraw and think things through on my own...'
    },
{
        'key': 'q5',
        'label': '5. What kind of work or activities make you feel most fulfilled?',
        'placeholder': 'e.g. I love solving complex problems and creating new ideas...'
    },
    {
        'key': 'q6',
        'label': '6. Describe how you organize your daily life and plans.',
        'placeholder': 'e.g. I always plan ahead and follow a strict schedule...'
    },
]

#Load model ột lần duy nhất (cache lại)
@st.cache_resource
def load_predictor():
    return MBTIPredictor()
predictor = load_predictor()

#giao diện
st.title("🧠 MBTI PERSONALITY PREDICTOR")
st.write("Answer the questions below in English. The more detail you write, the more accurate the prediction.")
st.write("~~~")

#Hiển thị từng câu hỏi
answers = {}
for q in QUESTIONS:
    answers[q['key']] = st.text_area(
        q['label'],
        placeholder=q['placeholder'],
        height=100,
        key=q['key']
    )
st.write("~~~")

if st.button("Predict My MBTI", type ="primary"):
    # Ghép tất cả câu trả lời lại
    combined = ' '.join([
        answers[q['key']] for q in QUESTIONS
        if answers[q['key']].strip()  # bỏ qua ô trống
    ])

    # Kiểm tra có đủ text không
    word_count = len(combined.split())
    if word_count < 30:
        st.warning(f"Please write more! Currently {word_count} words — need at least 30 words for accurate prediction.")
    else:
        result, details = predictor.predict(combined)

        # Hiển thị kết quả
        st.success(f"## Your MBTI: **{result}**")
        st.write(f"**{MBTI_DESC.get(result, '')}**")
        st.write(
            f"*(Based on {word_count} words across {sum(1 for q in QUESTIONS if answers[q['key']].strip())} answers)*")

        # 4 chiều
        st.write("Breakdown:")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("**Energy**")
            st.markdown("Introvert 🤫" if details['I_E'] == 'I' else "Extrovert 🗣️")

        with col2:
            st.markdown("**Perception**")
            st.markdown("Intuition 🔮" if details['N_S'] == 'N' else "Sensing 👁️")

        with col3:
            st.markdown("**Judgment**")
            st.markdown("Thinking 🧠" if details['T_F'] == 'T' else "Feeling ❤️")

        with col4:
            st.markdown("**Lifestyle**")
            st.markdown("Judging 📋" if details['J_P'] == 'J' else "Perceiving 🌊")
        # Hiển thị text đã ghép (cho người dùng kiểm tra)
        with st.expander("See combined text sent to model"):
            st.write(combined)