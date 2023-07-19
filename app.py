import openai
import pandas as pd
from openai.embeddings_utils import cosine_similarity
import numpy as np
import os
from tqdm import tqdm
import streamlit as st


openai.api_key = os.getenv("OPENAI_API_KEY")

df = pd.read_csv("fed-speech-embedded.csv")
df["embeddings"] = df["embeddings"].apply(eval).apply(np.array)

def get_embedded(word):
    embedded_word = openai.Embedding.create(
        model = "text-embedding-ada-002",
        input = word
    )
    return embedded_word["data"][0]["embedding"]


if __name__ == "__main__":
    st.title("AI Economics Teacher")
    aidesc = ""
    user_input = st.text_input("Ask a question:", placeholder="")
    results = st.empty()
    if st.button("Submit"):
        user_embedded = get_embedded(user_input)
        df["similarity"] = df["embeddings"].apply(lambda x: cosine_similarity(x, user_embedded))
        sorted_df = df.sort_values(by=["similarity"], ascending=False)
        five_results = []

        for i in range(5):
            five_results.append(sorted_df.iloc[i]["text"])
        results.write(five_results)
