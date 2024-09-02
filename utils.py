from openai import OpenAI
import re

def get_answer(query, system_prompt="", api_key=None):
    client = OpenAI(api_key=api_key)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    )
    return completion.choices[0].message.content

def contains_unicode_pattern(s):
    pattern = r'[\u0080-\uFFFF]+(?:\s[\u0080-\uFFFF]+)*'
    return re.search(pattern, s) is not None