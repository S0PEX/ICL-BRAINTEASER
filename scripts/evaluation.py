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
    return response


def model_response_to_letter_postprocessed(response: str) -> str:
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


def is_model_response_correct(result: ExecutionResult) -> tuple[bool, bool]:
    """
    Evaluate the correctness of a model answer by comparing it to the correct answer.

    Args:
        result (ExecutionResult): The result object containing the model output and the riddle

    Returns:
        tuple[bool, bool]: Tuple of raw correctness and postprocessed correctness
    """
    riddle_answer = result.riddle.answer
    riddle_answer_letter = string.ascii_uppercase[result.riddle.label]
    raw_model_answer = result.model_output.get_ai_response().content
    raw_model_answer = raw_model_answer.strip()

    # Raw accuracy check
    raw_correct = raw_model_answer.startswith(
        (
            riddle_answer,  # Starts with the answer (e.g., "The man is a barber")
            f"({riddle_answer_letter}) {riddle_answer}",  # Starts with the letter in parentheses followed by the answer
        )
    )
    raw_correct = (
        raw_correct
        or (
            # Only for cases where the model answer is shorter than the correct answer
            len(raw_model_answer) < len(riddle_answer)
            and raw_model_answer.startswith(
                (
                    riddle_answer_letter,  # Starts with the letter (e.g., "A")
                    f"({riddle_answer_letter})",  # Starts with the letter in parentheses (e.g., "(A)")
                )
            )
        )
    )

    # Postprocessed accuracy check
    postprocessed_model_answer = model_response_to_letter_postprocessed(
        raw_model_answer
    )
    postprocessed_correct = (
        riddle_answer_letter == postprocessed_model_answer
        or postprocessed_model_answer.startswith(riddle_answer)
        or raw_correct
    )

    return raw_correct, postprocessed_correct


def calculate_model_accuracy(
    results: list[ExecutionResult],
    debug_print: bool = False,
) -> tuple[float, float, float, float]:
    """
    Evaluate model results by comparing model answers to correct answers.

    Args:
        results (List[ExecutionResult]): List of execution results containing model outputs and riddles
        debug_print (bool, optional): Whether to print debugging information for incorrect answers. Defaults to False.

    Returns:
        float: Raw percentage of correct answers (0-100)
        float: Raw fraction of correct answers (0-1)
        float: Postprocessed percentage of correct answers (0-100)
        float: Postprocessed fraction of correct answers (0-1)
    """
    raw_correct_answers_list = []
    postprocessed_correct_answers_list = []

    for result in results:
        riddle_answer = result.riddle.answer
        riddle_answer_letter = string.ascii_uppercase[result.riddle.label]
        raw_model_answer = result.model_output.get_ai_response().content

        # Raw accuracy check
        raw_correct = raw_model_answer.startswith(
            (
                riddle_answer,  # Starts with the answer (e.g., "The man is a barber")
                f"({riddle_answer_letter}) {riddle_answer}",  # Starts with the letter in parentheses followed by the answer
            )
        )
        raw_correct = (
            raw_correct
            or (
                # Only for cases where the model answer is shorter than the correct answer
                len(raw_model_answer) < len(riddle_answer)
                and raw_model_answer.startswith(
                    (
                        riddle_answer_letter,  # Starts with the letter (e.g., "A")
                        f"({riddle_answer_letter})",  # Starts with the letter in parentheses (e.g., "(A)")
                    )
                )
            )
        )
        raw_correct_answers_list.append(raw_correct)

        # Postprocessed accuracy check
        postprocessed_model_answer = model_response_to_letter_postprocessed(
            raw_model_answer
        )
        postprocessed_correct = (
            riddle_answer_letter == postprocessed_model_answer
            or postprocessed_model_answer.startswith(riddle_answer)
            or raw_correct
        )
        postprocessed_correct_answers_list.append(postprocessed_correct)

        if not postprocessed_correct and debug_print:
            print(
                f"Model Answer: {postprocessed_model_answer} | Correct Answer: {riddle_answer_letter}"
            )

    # Calculate raw accuracy
    raw_correct_answers = sum(raw_correct_answers_list)
    total_answers = len(raw_correct_answers_list)
    raw_correct_answers_fraction = raw_correct_answers / total_answers
    raw_correct_answers_percentage = raw_correct_answers_fraction * 100

    # Calculate postprocessed accuracy
    postprocessed_correct_answers = sum(postprocessed_correct_answers_list)
    postprocessed_correct_answers_fraction = (
        postprocessed_correct_answers / total_answers
    )
    postprocessed_correct_answers_percentage = (
        postprocessed_correct_answers_fraction * 100
    )

    return (
        raw_correct_answers_percentage,
        raw_correct_answers_fraction,
        postprocessed_correct_answers_percentage,
        postprocessed_correct_answers_fraction,
    )
