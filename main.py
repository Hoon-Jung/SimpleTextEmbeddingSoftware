import openai
import pandas as pd
from openai.embeddings_utils import cosine_similarity
import numpy as np
import os
from tqdm import tqdm


openai.api_key = os.getenv("OPENAI_API_KEY")


# df = pd.read_csv("sample_words.csv")
df = pd.read_csv("fed-speech-embedded.csv")
df["embeddings"] = df["embeddings"].apply(eval).apply(np.array)

def get_embedded(word):
    embedded_word = openai.Embedding.create(
        model = "text-embedding-ada-002",
        input = word
    )
    return embedded_word["data"][0]["embedding"]

tqdm.pandas()
# df["embeddings"] = df["text"].progress_apply(lambda x: get_embedded(x))

# word1 = get_embedded("french fries")
# word2 = get_embedded("fizzy")

# sim = cosine_similarity(word1, word2)
# print(df["embeddings"])
# df.to_csv("fed-speech-embedded.csv", index=False)

search = "what is inflation?"
search = get_embedded(search)

df["similarity"] = df["embeddings"].progress_apply(lambda x: cosine_similarity(x, search))
sorted_df = df.sort_values(df["similarity"], ascending=False)

for i in range(5):
    print(sorted_df.iloc[i]["text"], sorted_df.iloc[i]["embeddings"])
# print(search)