import openai
from openai.error import RateLimitError
import backoff

@backoff.on_exception(backoff.expo, RateLimitError)
def ChatGPT(dialog, temperature=0.7, max_tokens=2048, model='gpt-3.5-turbo', functions = None, keep_response = False):
    assert isinstance(dialog, list)
    if functions is not None:
        response = openai.ChatCompletion.create(
            model=model,
            messages=dialog,
            temperature=temperature, 
            max_tokens=max_tokens,
            functions = functions
        )
    else:
        response = openai.ChatCompletion.create(
            model=model,
            messages=dialog,
            temperature=temperature, 
            max_tokens=max_tokens,
        )
    # except:
    #     if recursive:
    #         return ChatGPT(dialog=dialog, temperature=temperature, max_tokens=max_tokens, model=model, 
    #                     functions = functions, keep_response = keep_response, recursive=False)
    #     return None
    if keep_response:
        return response['choices'][0]['message']
    return response['choices'][0]['message']['content']


def get_dialog_string(gpt4_dialog, llama2_dialog):
    prompt_string = ""
    for r in gpt4_dialog:
        if r['role'] == "system":
            prompt_string += f"Start of INSTRUCTION: {r['content']}\nEnd of INSTRUCTION\n\n# Dialog between Assistant A and Student:"
        if prompt_string:
            prompt_string += "\n"

        if r['role'] == "user":
            prompt_string += f"STUDENT: {r['content']}"
        if r['role'] == "assistant":
            prompt_string += f"ASSISTANT A: {r['content']}"

    prompt_string += "\n\n# Dialog between Assistant B and Student:"

    for r in llama2_dialog:
        if prompt_string:
            prompt_string += "\n"

        if r['role'] == "user":
            prompt_string += f"STUDENT: {r['content']}"
        if r['role'] == "assistant":
            prompt_string += f"ASSISTANT B: {r['content']}"

    return prompt_string