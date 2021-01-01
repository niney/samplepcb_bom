import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text

model = None

def init():
    global model
    module_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
    model = hub.load(module_url)

def doc2vect(text):
    return embed_text(text)

def docs2vects(text_list):
    embed_text_list = []
    for text in text_list:
        embed_text_list.append(embed_text(text)[0])
    return embed_text_list

def embed_text(text):
    global model
    #vectors = session.run(embeddings, feed_dict={text_ph: text})
    vectors = model(text)
    return [vector.tolist() for vector in np.array(vectors)]
