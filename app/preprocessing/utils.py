import os
import numpy as np
import tensorflow as tf

async def save_file(audio_file):
    assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../assets"))
    file_path = os.path.join(assets_dir, audio_file.filename)

    with open(file_path, "wb") as f:
        f.write(await audio_file.read())

    return file_path

def normalize_result(results):
    normalized_results = []
    for x in range(tf.shape(results)[0]):
        curr_result = results[x]
        normalized_results.append([int(i) for i in np.where(curr_result > 0.6, 1, 0)])
    return normalized_results