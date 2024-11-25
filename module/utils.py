import re

def extract_answer(response):
    """
    extracts the answer from the response using a regular expression.
    expected format: "[ANSWER]: (A) convolutional networks"

    if there are any answers formatted like the format, it returns None.
    """
    pattern = r"\[ANSWER\]:\s*\((A|B|C|D|E)\)"
    match = re.search(pattern, response)

    if match:
        return match.group(1)
    else:
        return extract_again(response)

def extract_again(response):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, response)
    if match:
        return match.group(0)
    else:
        return None

def calculate_accuracy(df):
    df["Correct"] = df["answer"] == df["pred_answer"]
    accuracy = df["Correct"].mean() * 100
    return accuracy