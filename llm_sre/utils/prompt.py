from llm_sre.models.example import Example

NEW_LINE = "\n"


def parse_prompt(instruction_steps: list[str], prompt_input: str) -> str:
    return f"""
    {prompt_input}
    
    program:
    {NEW_LINE.join([f"- {instruction}" for instruction in instruction_steps])}
    - return your answer and begin your response with RESPONSE.
    
    State each step of the program and show your work for performing that step. 
    
    1. {instruction_steps[0]}
    """


def parse_prompt_with_examples(instruction_steps: list[str], examples: list[Example], prompt_input: str) -> str:
    return f"""
    {prompt_input}
    
    program:
    {NEW_LINE.join([f"- {instruction}" for instruction in instruction_steps])}
    - return your answer and begin your response with RESPONSE.
    
    examples:
    {NEW_LINE.join([example.get_text() for example in examples])}
    
    State each step of the program and show your work for performing that step. 
    
    1. {instruction_steps[0]}
    """


def parse_prompt_llama(examples: list[Example], prompt_input: str) -> str:
    return f"""Examples:\n{NEW_LINE.join([example.get_text() for example in examples])}\n- QUESTION: {prompt_input}"""
