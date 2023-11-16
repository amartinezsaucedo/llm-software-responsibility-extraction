from llm_sre.models.requirement import Requirement


def draw_responsibility_graphs(requirements: list[Requirement], output_folder_path: str):
    for requirement in requirements:
        requirement.save_graph(output_folder_path)
