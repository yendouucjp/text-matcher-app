import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🧠 高精度テキストマッチング（BERT使用）")
st.write("日本語の意味をベースにしたマッチングを行います。")

# モデル読み込み（初回は少し時間がかかります）
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# 入力
a_input = st.text_area("✏️ A群のテキスト（1行1テキスト）", height=200)
b_input = st.text_area("✏️ B群のテキスト（1行1テキスト）", height=200)

top_n = st.slider("各Aテキストに対して表示するBテキストの数", 1, 5, 1)

if st.button("✅ マッチング実行"):
    group_a = [line.strip() for line in a_input.split("\n") if line.strip()]
    group_b = [line.strip() for line in b_input.split("\n") if line.strip()]
    
    if not group_a or not group_b:
        st.warning("A群・B群のテキストをそれぞれ1つ以上入力してください。")
    else:
        emb_a = model.encode(group_a)
        emb_b = model.encode(group_b)

        sim_matrix = cosine_similarity(emb_a, emb_b)

        matches = []
        for i, row in enumerate(sim_matrix):
            top_matches = row.argsort()[::-1][:top_n]
            for j in top_matches:
                matches.append({
                    "A_index": i,
                    "A_text": group_a[i],
                    "B_index": j,
                    "B_text": group_b[j],
                    "類似度": round(row[j], 3)
                })

        result_df = pd.DataFrame(matches)
        st.success("✅ マッチング完了！（意味ベース）")
        st.dataframe(result_df)
