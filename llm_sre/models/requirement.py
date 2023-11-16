from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
from llm_sre.models.responsibility import Responsibility


class Requirement:
    _id: int
    _text: str
    _responsibilities: list[Responsibility]
    _responsibilities_sequence: nx.DiGraph
    _sentences: list[str]

    def __init__(self, requirement_id: int, text: str):
        self._id = requirement_id
        self._text = text
        self._responsibilities = []
        self._responsibilities_sequence = nx.DiGraph()
        self._sentences = []

    def get_text(self) -> str:
        return self._text

    def get_sentences(self) -> list[str]:
        return self._sentences

    def get_sentence_by_index(self, index: int) -> str:
        return self._sentences[index]

    def add_responsibilities(self, responsibilities: list[tuple[list[str], int]]):
        for responsibility in responsibilities:
            responsibility_texts = responsibility[0]
            sentence_id = responsibility[1]
            for text in responsibility_texts:
                new_responsibility = Responsibility(len(self._responsibilities) + 1, text, sentence_id)
                self.add_responsibility(new_responsibility)

    def add_responsibility(self, responsibility: Responsibility):
        if not any(r.is_responsibility(responsibility.get_text()) for r in self._responsibilities):
            self._responsibilities.append(responsibility)
            self._responsibilities_sequence.add_node(responsibility)

    def get_responsibilities(self, sentence_index: int = None) -> list[Responsibility]:
        if sentence_index:
            return list(filter(lambda responsibility: responsibility.get_sentence_id() == sentence_index,
                               self._responsibilities))
        return self._responsibilities

    def add_causal_relationship(self, relationship_structure: list[str], sentence_index: int,
                                add_responsibilities_if_missing: bool = False):
        if len(relationship_structure) == 2:
            cause = self.find_responsibility(relationship_structure[0], sentence_index, add_responsibilities_if_missing)
            effect = self.find_responsibility(relationship_structure[1], sentence_index,
                                              add_responsibilities_if_missing)
            if cause and effect:
                self._responsibilities_sequence.add_edge(cause, effect)
        else:
            print(relationship_structure)

    def find_responsibility(self, responsibility_text: str, sentence_index: int = 0,
                            add_if_missing: bool = False) -> Responsibility:
        responsibility = next((responsibility for responsibility in self._responsibilities if
                               responsibility.is_responsibility(responsibility_text)), None)
        if not responsibility and add_if_missing:
            responsibility = Responsibility(len(self._responsibilities) + 1, responsibility_text, sentence_index)
            self.add_responsibility(responsibility)
        return responsibility

    def save_graph(self, output_folder_path: str):
        nx.draw_networkx(self._responsibilities_sequence, with_labels=True)
        plt.savefig(Path(output_folder_path) / f"requirement_{self._id}.png")
        plt.clf()

    def get_summary(self) -> str:
        return f"""\n@Requirement=REQ-{self._id}\n{self._text}\n@Responsibilities={len(self._responsibilities)}\n{self._get_responsibilities()}"""

    def _get_responsibilities(self) -> str:
        return "\n".join(
            [f"Resp [{self._id}-{responsibility.get_id()}] {responsibility.get_text()}" for responsibility in
             self._responsibilities])

    def remove_responsibility(self, responsibility: Responsibility):
        self._responsibilities = list(
            filter(lambda r: r.get_text() != responsibility.get_text(), self._responsibilities))

    def set_sentences(self, sentences: list[str]):
        self._sentences = sentences

    def get_edges(self):
        return ", ".join([f"{edge[0]}->{edge[1]}" for edge in self._responsibilities_sequence.edges])
