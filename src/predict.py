import os
import joblib
from preprocess import clean_text

DIMENSIONS = ['I_E', 'N_S', 'T_F', 'J_P']

# Ánh xạ từ nhãn 0/1 sang ký tự MBTI cho từng chiều
LABEL_MAP = {
    'I_E': {1: 'I', 0: 'E'},
    'N_S': {1: 'N', 0: 'S'},
    'T_F': {1: 'T', 0: 'F'},
    'J_P': {1: 'J', 0: 'P'},
}

class MBTIPredictor:
     def __init__(self):
         base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
         models_dir = os.path.join(base_dir, 'models')

         #load vectorizer
         self.vectorizer = joblib.load(
             os.path.join(models_dir, 'vectorizer.pkl')
         )
         #Load 4 model tốt nhất
         self.models = {}
         for dim in DIMENSIONS:
             path = os.path.join(models_dir, f'best_{dim}.pkl')
             self.models[dim] = joblib.load(path)

     def predict(self, text):
         #Làm sạch text
         cleaned = clean_text(text)
         #Chuyển thành TF-IDF
         X = self.vectorizer.transform([cleaned])
         #Chạy 4 model và ghép kết quả
         result = ''
         details = {}
         for dim in DIMENSIONS:
             pred = self.models[dim].predict(X)[0]
             letter = LABEL_MAP[dim][pred]
             result += letter
             details[dim] = letter
         return result, details

if __name__ == '__main__':
    predictor = MBTIPredictor()

    # Test thử với vài đoạn text
    '''
    test_texts = [
        "I love spending time alone reading philosophy books and thinking about the meaning of life. I prefer deep conversations over small talk.",
        "Let's party tonight! I love meeting new people and being the center of attention. Logic and facts matter most to me.",
    ]

    for text in test_texts:
        result, details = predictor.predict(text)
        print(f"\nText: {text[:60]}...")
        print(f"MBTI dự đoán: {result}")
        print(f"Chi tiết: {details}")
    '''
    print("DỰ ĐOÁN MBTI TỪ VĂN BẢN")
    print("Nhập đoạn text (tiếng Anh), gõ 'quit' để thoát")

    while True:
        text = input("\nNhập text: ")

        if text.lower() == 'quit':
            print("Tạm biệt!")
            break

        if len(text.strip()) == 0:
            print("Vui lòng nhập text!")
            continue

        result, details = predictor.predict(text)
        print(f"MBTI dự đoán: {result}")
        print(f"Chi tiết: {details}")
