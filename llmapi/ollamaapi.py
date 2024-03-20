import ollama
import numpy as np
import time


def get_response(prompt: str):
    return ollama.chat(model='vicuna', messages=[{
        'role': 'user',
        'content': prompt,
    },
    ])['message']['content']


if __name__ == "__main__":
    time_list = []
    for _ in range(30):
        t1 = time.time_ns()
        print(get_response("Who is barack obama"))
        t2 = time.time_ns()
        time_list.append((t2-t1) / 1e9)

    print(np.average(time_list))