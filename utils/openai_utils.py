import openai
from openai.error import RateLimitError
import backoff

@backoff.on_exception(backoff.expo, RateLimitError)
def ChatGPT(dialog, temperature=0.7, max_tokens=2048, model='gpt-3.5-turbo'):
    assert isinstance(dialog, list)

    response = openai.ChatCompletion.create(
        model=model,
        messages=dialog,
        temperature=temperature, 
        max_tokens=max_tokens
    )

    return {
        "response": response['choices'][0]['message']['content'],
        "usage": response['usage']
    }


def convert_dialogs(dialog_a, dialog_b):
    prompt_string = ""
    for r in dialog_a:
        if r['role'] == "system":
            prompt_string += f"Start of INSTRUCTION: {r['content']}\nEnd of INSTRUCTION\n\n# Dialog between Assistant A and Student:"
        if prompt_string:
            prompt_string += "\n"

        if r['role'] == "user":
            prompt_string += f"STUDENT: {r['content']}"
        if r['role'] == "assistant":
            prompt_string += f"ASSISTANT A: {r['content']}"

    prompt_string += "\n\n# Dialog between Assistant B and Student:"

    for r in dialog_b:
        if prompt_string:
            prompt_string += "\n"

        if r['role'] == "user":
            prompt_string += f"STUDENT: {r['content']}"
        if r['role'] == "assistant":
            prompt_string += f"ASSISTANT B: {r['content']}"

    return prompt_string

def Compare2Dialog(dialog_a, dialog_b, answer_choices, model):
    prompt_string = convert_dialogs(dialog_a, dialog_b)
    if isinstance(answer_choices, list):
        answer_choices = ", ".join(answer_choices)

    dialog = [
        {"role": "system", "content" :f"""You are going to analyze the performance of two ASSISTANTs below. Basically, the ASSISTANT is asked to role-play a virtual patient, and answer the question from medical STUDENT. INSTRUCTION is the instruction given to ASSISTANT, and it includes details about that case. 
        Criteria:
        - Who follow the INSTRUCTION better? (assistant A or B).
        
        {prompt_string}"""},
        {"role": "user", "content": f"Answer choices: {answer_choices}. Only answer directly, no explanation."}
    ]

    package = ChatGPT(
        dialog=dialog,
        model=model,
        temperature=0.3, 
        max_tokens=512
    )

    return package