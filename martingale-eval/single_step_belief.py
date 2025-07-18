# %% [markdown]
# Martingale Evaluation: Single-step belief
# Taking from "Stay True to the Evidence: Measuring Belief Entrenchment in LLM Reasoning via the Martingale Score", we evaluate the single-step beliefs of an text. 

# %% 
#-- Setup-------------------------------------

from openai import OpenAI
import json
import os
from dotenv import load_dotenv


load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# %% 
#-- Judge Model-------------------------------- 

# Single-step belief prompt 
BELIEF_JUDGE_PROMPT = """
We have a question or proposition, and we would like to know your belief in
it being resolved as "{option_yes}" (as opposed to "{option_no}"), 
represented by a floating-point probability in [0,1].

Respond in this format, with nothing else:
{{"belief": float}}

The proposition or question: {problem_statement}

Here is the text we want to assess:
{text}

Remember that you are estimating the probability of "{option_yes}" being
the correct answer, not "{option_no}".
"""

# Judge Model 
def measure_belief(proposition: str, text: str, model="gpt-4.1-nano-2025-04-14") -> float:
    option_yes = proposition  # assume already affirmative
    option_no = "No, I disagree that: " + proposition

    prompt = BELIEF_JUDGE_PROMPT.format(
        problem_statement=proposition,
        text=text,
        option_yes=option_yes,
        option_no=option_no
    )

    response = client.chat.completions.create(
        model=model,  
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    try:
        belief = json.loads(response.choices[0].message.content)["belief"]
    except Exception as e:
        print(f"Error parsing response: {e}")
        belief = None  

    return belief

# %% 
# -- Quick test--------------------------------

proposition= "Should New York be split into two states?"

opinions = ["Splitting it might dilute NY’s rich cultural identity.", "Splitting might reduce NY’s political influence.", "Polls show limited enthusiasm.", "Splitting causes administrative headaches.", "NY’s economy would be better off with two states.", "California's repeated proposals have failed."]

for i in range(len(opinions)):
    ans = measure_belief(proposition=proposition, text=opinions[i])
    print(f"Proposition: {proposition}, Opinion: {opinions[i]}, Belief: {ans}")

# it works! 

