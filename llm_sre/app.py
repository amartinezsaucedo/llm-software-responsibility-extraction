from llm_sre.responsibility_extractor import extract_responsibilities_from_file
from llm_sre.responsibility_sequencer import sequentialize_responsibilities
from llm_sre.responsibility_analyzer import draw_responsibility_graphs

if __name__ == "__main__":
    requirements = extract_responsibilities_from_file("./cases/example.txt")
    sequentialize_responsibilities(requirements)
    draw_responsibility_graphs(requirements, "./cases")
