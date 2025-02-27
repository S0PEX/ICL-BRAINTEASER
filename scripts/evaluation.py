import re
import string

from scripts.executor import ExecutionResult


def model_response_to_letter(response: str) -> str:
    """
    Extract the answer letter (A, B, C, D) from various response formats.

    Args:
        response (str): The raw response string from the model

    Returns:
        str: The extracted answer letter or the original response if no pattern matches
    """
    # Strip any leading/trailing whitespace or newlines
    response = response.strip()

    # Check if the response is long enough to avoid index out of bounds
    if len(response) < 2:
        return response

    # Case 0: Check for letter in parentheses anywhere in the response (e.g., "(A)")
    regex_in_braces_somewhere = re.search(r"\((A|B|C|D)\)", response)
    if regex_in_braces_somewhere:
        return regex_in_braces_somewhere.group(1)

    # Case 1: If the response starts with a letter followed by a closing parenthesis (e.g., A), (B), etc.)
    if response[1] == ")":
        return response[0]

    # Case 2: If the response starts with "Answer: " (e.g., Answer: A, Answer: B)
    elif response.startswith("Answer: "):
        # Ensure the length is enough to extract the letter after "Answer: "
        if len(response) > 8:
            return response[8:9]  # Extracts the letter right after "Answer: "

    # Case 3: If the response starts with "**" (e.g., **A, **B)
    elif response.startswith("**"):
        # Ensure the length is enough to extract the letter after "**"
        if len(response) > 2:
            return response[2:3]  # Extracts the letter after "**"

    # Case 4: For other formats where the letter might be the only thing
    # (e.g., A., B., C. etc.)
    elif response[1] == "." or response[1] == "\n":
        return response[0]
    # Case 5: If there's a space before the letter (e.g., " A", " B", etc.)
    elif len(response) > 1 and response[0] == " " and response[1].isalpha():
        return response[1]  # Extract the letter after the space
    # Case 6: If there's a space after the letter and then new line (e.g., "A \n", "B \n", etc.)
    elif (
        len(response) > 2
        and response[0].isalpha()
        and response[1] == " "
        and "\n" in response[1:5]
    ):
        return response[0]  # Extract the letter after the space
    # Default: If no pattern matches, return the original response
    else:
        return response


def eval_model_results(
    results: list[ExecutionResult], debug_print: bool = False
) -> float:
    """
    Evaluate model results by comparing model answers to correct answers.

    Args:
        results (List[ExecutionResult]): List of execution results containing model outputs and riddles
        debug_print (bool, optional): Whether to print debugging information for incorrect answers. Defaults to False.

    Returns:
        float: Percentage of correct answers (0-100)
    """
    correct_answers_list = []

    for result in results:
        riddle_answer = result.riddle.answer
        riddle_answer_letter = string.ascii_uppercase[result.riddle.label]
        model_answer = result.model_output.get_ai_response().content
        model_answer = model_response_to_letter(model_answer)

        # Check if the answer is correct (either matching the letter or starting with the answer text)
        correct = riddle_answer_letter == model_answer or model_answer.startswith(
            riddle_answer
        )

        if not correct and debug_print:
            print(
                f"Model Answer: {model_answer} | Correct Answer: {riddle_answer_letter}"
            )

        correct_answers_list.append(correct)

    correct_answers = sum(correct_answers_list)
    total_answers = len(correct_answers_list)
    correct_answers_percentage = correct_answers / total_answers * 100
    return correct_answers_percentage
