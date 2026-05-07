import streamlit as st
import numpy as np
import os
import cv2
from paddleocr import PaddleOCR
from PIL import Image
from groq import Groq

# 1. 初始化 PaddleOCR (建議加上 cache)
@st.cache_resource
def load_ocr_model():
    # lang='ch' 代表支援中英文，use_angle_cls 幫助識別旋轉了的收據
    return PaddleOCR(use_angle_cls=True, lang='ch')

# 2. 初始化 Groq LLM
client = Groq(os.environ.get("GROQ_API_KEY"))

def main():
    st.set_page_config(page_title="AI Receipt Scanner", layout="centered")
    st.title("🧾 商場級收據智能識別")
    st.write("結合 PaddleOCR 精度與 LLM 語意分析")

    # 影相功能
    img_file = st.camera_input("請對準收據拍攝")

    if img_file:
        # 將 Streamlit 檔案轉為 OpenCV 格式
        image = Image.open(img_file)
        img_array = np.array(image)
        # PaddleOCR 需要 BGR 格式
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        with st.spinner("PaddleOCR 正在掃描文字..."):
            ocr = load_ocr_model()
            result = ocr.ocr(img_cv, cls=True)

            # 3. 提取所有文字並保留大概排版
            full_text = ""
            for idx in range(len(result)):
                res = result[idx]
                if res: # 確保有識別到內容
                    for line in res:
                        # line[1][0] 是文字內容, line[0] 是座標
                        full_text += f"{line[1][0]} "

            st.subheader("掃描結果")
            st.text_area("OCR 原始內容 (Raw Text):", full_text, height=150)

        # 4. 傳送給 LLM 進行結構化提取
        if full_text:
            with st.spinner("AI 正在分析數據..."):
                prompt = f"""
                以下是從收據 OCR 掃描出來的原始文字。請幫我分析並提取以下資訊：
                1. 商店名稱 (store_name)
                2. 日期 (date, YYYY-MM-DD)
                3. 總金額 (total_amount, 只要數字)
                4. 消費清單 (items, 包含品名與價格)

                原始文字：
                {full_text}

                請只以 JSON 格式輸出，不要有額外文字說明。
                """

                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile",
                    temperature=0
                )

                final_json = chat_completion.choices[0].message.content
                st.subheader("✅ 提取完成")
                st.json(final_json)

if __name__ == "__main__":
    main()