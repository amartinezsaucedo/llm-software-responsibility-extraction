# Llama 2 for software responsibility extraction
# Installation
1. Download desired LLM binaries and place it on `/models` folder.
2. If you choose to use Docker, simply run `docker compose up`. Append `--build` flag when running project for the first time. No more steps are needed.
3. Otherwise, depending on host capabilities hardware acceleration is available. For example, this repository uses Llama 2 as main LLM. Therefore, the steps to support hardware acceleration are listed on [Python Bindings for llama.cpp](https://github.com/abetlen/llama-cpp-python). Check `/notebooks` for examples.
4. Install dependencies running `poetry install`

# Run
```
python llm_sre/app.py [-h] -n NAME [-f FILE] -t {extract,sequence,export} [-c CHAT] -o OUTPUT [-a ADD_RESPONSIBILITIES] [-e EVALUATE]

options:
-n NAME, --name NAME          Project name
-f FILE, --file FILE          Requirements file
-m MODEL, --model MODEL       Model file path
-t {extract,sequence,export}  Task to execute
-c CHAT, --chat CHAT          Flag to indicate whether the LLM is chat based
-o OUTPUT, --output OUTPUT    Output folder to save results
-a ADD_RESPONSIBILITIES       Flag to indicate whether to add responsibilities in other steps
-e EVALUATE                   Flag to indicate whether to evaluate results
```
Example
```bash
python llm_sre/app.py -n MsLite -f ./cases/MsLite.txt -t extract -o ./output/MsLite/llama_no_add -m ./model/gguf-model-chat.bin
```
