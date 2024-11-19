import json


def wrap_prompt_completion(prompt, model_name: str = "text-davinci-003", max_tokens: int = 256, temperature: float = 0.0,
                top_p: float = 1.0):
    ret_dict = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "n": 1,
        "stream": False,
        "logprobs": 5,
        "stop": "\n"
    }
    return ret_dict


def wrap_prompt_chat(prompt, model_name: str = "text-davinci-003", max_tokens: int = 256, temperature: float = 0.0,
                top_p: float = 1.0):
    messages = [
        {'role': 'system', 'content': 'You are a helpful and concise assistant.'},
        {'role': 'user', 'content': prompt}
    ]
    ret_dict = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    return ret_dict


def process_list_from_output(ret_text):

    ret_list = []
    ret_lines = ret_text.split('\n')
    ret_lines = [x.split('(')[0] for x in ret_lines]
    assert len(ret_lines) < 30
    for lidx, line in enumerate(ret_lines):
        line = line.strip(' ').rstrip('\n')
        if len(line) < 3:
            continue
        if line[:2].isdigit():
            if line[2] != '.':
                continue
            else:
                line = line[3:].lstrip(' ')
                ret_list.append(line)
        elif line[0].isdigit():
            if line[1] != '.':
                continue
            else:
                line = line[2:].lstrip(' ')
                ret_list.append(line)
        elif lidx == 0:
            ret_list.append(line.lstrip(' '))
        else:
            print(f"Discarded line: {line}")

    ret_list = [x.lower() for x in ret_list]
    return ret_list