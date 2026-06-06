import re
import sys

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet = True)
nltk.download('wordnet', quiet= True)
nltk.download('omw-1.4', quiet = True)

MBTI_TYPES = [
    'infp','infj','intp','intj',
    'isfp','isfj','istp','istj',
    'enfp','enfj','entp','entj',
    'esfp','esfj','estp','estj'
]
#Khởi tạo object và gán biến
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    text = text.replace('|||', ' ') #tác môi bài đăng trong 1 ô thành 1 oạn văn
    text = re.sub(r'http\S+|www\S+', '',text) #xóa url
    text = re.sub(r'<.*?>', '', text)#xóa html tags
    for t in MBTI_TYPES:
        text = re.sub(t, '', text, flags=re.IGNORECASE) #xóa 16 tên nhóm MBTI trong bài đăng
    text = re.sub(r'[^a-zA-Z\s]', '', text) #xóa mọi thứ không phải chữ cái
    text = text.lower()
    tokens = text.split() #cắt chuỗi thành danh sách các từ
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words] #vòng lặp lọc stopwords và lemmatize ồng thời
    return ' '.join(tokens)

#Tạo 4 nhãn nhị phân
def create_binary_labels(df: pd.DataFrame) -> pd.DataFrame:
    df['I_E'] = df['type'].apply(lambda x:1 if x[0] == 'I' else 0)
    df['N_S'] = df['type'].apply(lambda x:1 if x[1] == 'N' else 0)
    df['T_F'] = df['type'].apply(lambda x:1 if x[2] == 'T' else 0)
    df['J_P'] = df['type'].apply(lambda x:1 if x[3] == 'J' else 0)
    return df

def run_preprocessing(input_path: str, output_path: str):
    print("Đang đọc dữ liệu")
    df = pd.read_csv(input_path)

    print("Đang làm sạch text")
    df['cleaned_posts'] = df['posts'].apply(clean_text)

    print("Đang tạo nhãn nhị phân")
    df = create_binary_labels(df)

    df.to_csv(output_path, index=False)
    print(f"đã lưu vào {output_path}")

if __name__ == '__main__':
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_preprocessing(
        input_path=os.path.join(base_dir, 'data', 'mbti_1.csv'),
        output_path=os.path.join(base_dir, 'data', 'mbti_processed.csv')
    )