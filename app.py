
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🧠 高精度テキストマッチング（BERT 使用）")
st.write("A群とB群のテキスト間の意味的な類似性に基づいてマッチングを行います。")

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

a_input = st.text_area("✏️ A群のテキスト（1行1テキスト）", height=200)
b_input = st.text_area("✏️ B群のテキスト（1行1テキスト）", height=200)
top_n = st.slider("各Aテキストに対して表示するBテキストの数", 1, 5, 1)

def highlight_similarity(val):
    if val >= 0.8:
        return "background-color: lightgreen"
    elif val >= 0.5:
        return "background-color: khaki"
    else:
        return "background-color: lightgray"

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
        result_df = result_df.sort_values(by="類似度", ascending=False)
        styled_df = result_df.style.applymap(highlight_similarity, subset=["類似度"])

        st.success("✅ マッチング完了！（意味ベース）")
        st.dataframe(styled_df)
