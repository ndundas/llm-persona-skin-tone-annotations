# llm-persona-skin-tone-annotations
---
This repository contains code and data for our study examining whether large language models (LLMs) can reproduce diverse human social perspectives in skin tone annotation through persona-framed prompt engineering. Using 72 synthetic faces and four LLMs (GPT-4, Claude, Llama, Gemini), we show how persona-framed prompts systematically shift Monk Skin Tone ratings, reveal low inter-model agreement, and highlight both opportunities and risks for scalable fairness auditing in healthcare AI.  

All scripts and analyses are provided to ensure full reproducibility of the results presented in the manuscript.
---

## Setting Up the Environment

Follow the steps below to set up the project and run the scripts:

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### Step 2: Create a Virtual Environment
```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Requirements
``` bash
pip install -r requirements.txt
```

--- 

## Running the Scripts 

### Option 1: Generate your own Skin Tone Ratings using the same pipeline 
Runs the API script to process the images and generte a CSV with the scores. Make sure to update paths, prompts, API keys. 
``` bash
python multillm_API_calls.py
```

### Option 2: Repeat anlaysis  
Run the LLM_Annotation_Analysis.ipynb that will regenerate all the anaylsis from the figure. If you change the file path for the data annotations you can run the analysis on your personal results. 

--- 
## Notes and Considerations

### 1. ALL API Key in env 
Ensure that you set your all your API keys as an environemt variable in the main.py script. To set up your key, go to the OpenAI API Key Page and sign in or create an account. Next, navigate to the API Keys section in the OpenAI dashboard and click Create new secret key and copy the generated API key. This is a secret key—do not share it publicly! Repeat for all models you would like to test. 
### 2. Adaptability
These scripts dynamically handel additional prompts. If you want to add more prompts you can add directly in the multillm_API_calls.py to test them out on the same images.
