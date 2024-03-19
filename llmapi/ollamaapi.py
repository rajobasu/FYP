import ollama


def get_response(prompt: str):
    ollama.pull("vicuna")
    return ollama.chat(model='vicuna', messages=[{
        'role': 'user',
        'content': prompt,
    },
    ])['message']['content']

if __name__ == "__main__":
    print(get_response("Who is barack obama"))
