from llm_sre.models.example import Example


def load_examples(lines: list[str], human_prefix: str = "Human:", ai_prefix: str = "AI:") -> list[Example]:
    examples: list[Example] = []
    current_index = lines.index(human_prefix)
    while current_index < len(lines):
        ai_index = lines.index(ai_prefix, current_index+1)
        try:
            next_human_index = lines.index(human_prefix, current_index+1)
        except ValueError:  # If it throws ValueError it means this Human index is last conversation
            next_human_index = len(lines)
        prompt = "\n".join(lines[current_index+1:ai_index])
        answer = "\n".join(lines[ai_index+1:next_human_index])
        examples.append(Example(prompt, answer))
        current_index = next_human_index
    return examples


class TaskConfiguration:
    SYSTEM_MESSAGE = "System message:"
    PROMPT_TEMPLATE = "Template:"
    EXAMPLES = "Examples:"
    _system_message: str
    _prompt_template: str
    _examples: list[Example]

    def __init__(self, filename: str):
        with open(filename) as file:
            lines = [line.rstrip() for line in file]
            system_message_index = lines.index(self.SYSTEM_MESSAGE)
            prompt_template_index = lines.index(self.PROMPT_TEMPLATE)
            examples_index = lines.index(self.EXAMPLES)
            self._system_message = "\n".join(lines[system_message_index+1:prompt_template_index])
            self._prompt_template = "\n".join(lines[prompt_template_index+1:examples_index])
            self._examples = load_examples(lines[examples_index:])

    def get_system_message(self) -> str:
        return self._system_message

    def get_prompt_template(self) -> str:
        return self._prompt_template

    def get_examples(self) -> list[Example]:
        return self._examples
