import pickle
import pandas as pd

from llm_sre.models.requirement import Requirement


def save_object(filename: str, folder: str, data: object):
    with open(f"{folder}/{filename}.pkl", 'rb') as out:
        pickle.dump(data, out, pickle.HIGHEST_PROTOCOL)


def open_file(filename: str, folder: str):
    with open(f"{folder}/{filename}.pkl", 'rb') as inp:
        return pickle.load(inp)


def save_experiment(requirements: list[Requirement], filename: str, folder: str):
    dataframe_responsibilities = pd.DataFrame(
        [{"requirement": requirement.get_text(), "responsibilities": requirement.get_responsibilities()} for requirement
         in requirements])
    dataframe_sequences = pd.DataFrame(
        [{"requirement": requirement.get_text(), "sequences": requirement.get_edges()} for requirement
         in requirements])
    with pd.ExcelWriter(f"{folder}/{filename}.xlsx") as writer:
        dataframe_responsibilities.to_excel(writer, sheet_name='Responsibilities')
        dataframe_sequences.to_excel(writer, sheet_name='Sequences')
