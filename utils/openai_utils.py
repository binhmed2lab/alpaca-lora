import openai
from openai.error import RateLimitError
import backoff
import re

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

def process_analysis(text, criterias):

    answer_choices = ["Assistant A", "Assistant B", "Equally Good", "Equally Bad"]
    criteria = [{"criteria": c} for c in criterias]
    for line in text.split("\n"):
        if line.strip() == "": continue
        try:
            idx = re.findall("^\d+", line)[0]
            idx = int(idx)
            answer_choice, explanation = line.split(" - ")
            answer_choice = answer_choice.strip()
            explanation = explanation.strip()
            for choice in answer_choices:
                if choice in answer_choice:
                    answer_choice = choice
                    break

            criteria[idx-1]['answer_choice'] = answer_choice
            criteria[idx-1]['explanation'] = explanation
        except Exception as e:
            pass

    return criteria


def Compare2Dialog(dialog_a, dialog_b, answer_choices, criterias, model):
    prompt_string = convert_dialogs(dialog_a, dialog_b)
    if isinstance(answer_choices, list):
        answer_choices = ", ".join(answer_choices)
    
    criterias_str = "\n".join(f"{idx+1}. {c}" for idx, c in enumerate(criterias))

    dialog = [
        {"role": "system", "content" :f"""You are going to analyze the performance of two ASSISTANTs below. Basically, the ASSISTANT is asked to role-play a virtual patient, and answer the question from medical STUDENT. INSTRUCTION is the instruction given to ASSISTANT, and it includes details about that case. 
        
        {prompt_string}"""},
        {"role": "user", "content": f"""
        Criterias:
        {criterias_str}

        Answer choices: Assistant A, Assistant B, Equally Good, Equally Bad.
        Answer format: 
        1. Answer choice - Brief Explanation provides evidences and citations from their speech.
         
        Example Answer:
        1. Assistant A - Because...
        2. Assistant B - Because...
        3. Equally Bad - Because
        """}
    ]

    package = ChatGPT(
        dialog=dialog,
        model=model,
        temperature=0.3, 
        max_tokens=512
    )

    analysis_package = process_analysis(package, criterias)['response']

    return {
        "analysis": analysis_package,
        "usage": package['usage']
    }


