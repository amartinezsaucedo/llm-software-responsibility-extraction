import argparse

from llm_sre.tasks import AVAILABLE_TASKS
from llm_sre.tasks.io import open_file, save_experiment, save_object
from llm_sre.tasks.responsibility_extractor import extract_responsibilities_from_file
from llm_sre.tasks.responsibility_sequencer import sequentialize_responsibilities
from llm_sre.tasks.responsibility_analyzer import draw_responsibility_graphs


parser = argparse.ArgumentParser("Responsibility extraction using LLM")
parser.add_argument("-n", "--name", help="Project name", required=True)
parser.add_argument("-f", "--file", help="Requirements file")
parser.add_argument("-t", "--task", help="Task to execute", choices=AVAILABLE_TASKS, type=str, required=True)
parser.add_argument("-o", "--output", help="Output folder to save results", type=str, required=True)
parser.add_argument("-a", "-add_responsibilities", help="Flag to indicate whether to add responsibilities in other "
                                                        "steps", type=bool, default=False)
args = parser.parse_args()

task = args.task
name = args.name
file = args.file
output_folder = args.output
add_responsibilities = args.add_responsibilities

if task == "extract":
    requirements = extract_responsibilities_from_file(file)
    save_object(name, output_folder, requirements)

elif task == "sequence":
    requirements = open_file(name, output_folder)
    sequentialize_responsibilities(requirements, add_responsibilities)

elif task == "export":
    requirements = open_file(name, output_folder)
    draw_responsibility_graphs(requirements, output_folder)
    save_experiment(requirements, name, output_folder)

