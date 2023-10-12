from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
from llm_sre.models.responsibility import Responsibility


class Requirement:
    _id: int
    _text: str
    _responsibilities: list[Responsibility]
    _responsibilities_sequence: nx.DiGraph

    def __init__(self, requirement_id: int, text: str):
        self._id = requirement_id
        self._text = text
        self._responsibilities = []
        self._responsibilities_sequence = nx.DiGraph()

    def get_text(self) -> str:
        return self._text

    def add_responsibilities(self, responsibilities: list[str]):
        for responsibility_id, responsibility in enumerate(responsibilities):
            new_responsibility = Responsibility(responsibility_id, responsibility.strip())
            self._responsibilities.append(new_responsibility)
            self._responsibilities_sequence.add_node(new_responsibility)

    def get_responsibilities(self) -> list[Responsibility]:
        return self._responsibilities
    
    def add_causal_relationship(self, relationship_structure: list[str]):
        cause = self._find_responsibility(relationship_structure[0])
        effect = self._find_responsibility(relationship_structure[1])
        if cause and effect:
            self._responsibilities_sequence.add_edge(cause, effect)

    def _find_responsibility(self, responsibility_text: str) -> Responsibility:
        return next(responsibility for responsibility in self._responsibilities if responsibility.is_responsibility(responsibility_text))
    
    def save_graph(self, output_folder_path: str):
        nx.draw_networkx(self._responsibilities_sequence, with_labels=True)
        plt.savefig(Path(output_folder_path) / f"requirement_{self._id}.png")
        plt.clf()