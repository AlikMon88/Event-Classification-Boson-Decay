# A3: HEP Coursework

This repository contains my implementation of A3 Courework.

Please download the data for **Section B2** from the following link:  
[https://mkenzie.web.cern.ch/mkenzie/mphil/assignment/2425/Bs2DsPi](https://mkenzie.web.cern.ch/mkenzie/mphil/assignment/2425/Bs2DsPi)

Please download the data for **Section C2** from the following link:  
[https://mkenzie.web.cern.ch/mkenzie/mphil/assignment/2425/FCC](https://mkenzie.web.cern.ch/mkenzie/mphil/assignment/2425/FCC)

---

## Repository Structure

```plaintext
.
|
├── saves/                         # Contains the saved models/plots
|   
├── report/                        # Contains the coursework report
│   └── A3_CW.pdf
│
├── utils/                         # Python utility scripts for modularization
│   └── data_create.py             # Data Load and Prepare
│   └── train.py                   # Model Trainer
│   └── model.py                   # Contains Model (DNN, RNN, Deepsets)
│   
├── section_a.ipynb                # py notebook solution implementation for Section A2
├── section_b.ipynb                # py notebook solution implementation for Section B2
├── section_c_b.ipynb              # py notebook solution implementation for Section C2|
│
├── .gitignore                     # Git ignore file
├── README.md                      # This file
└── requirements.txt               # Python dependencies
```

# Setup Instructions
Follow the steps below to set up and run the project on your local machine:

## 1. Clone the Repository
```bash
git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a3_coursework/am3353.git
cd am3353
```

## 2. Set Up the Python Environment
Create and activate a venv
```bash 
# Create a virtual environment
python -m venv a3-env

# Activate the virtual environment
# On Windows:
a3-env\Scripts\activate
# On MacOS/Linux:
source a3-env/bin/activate
```

## 3. Install Dependencies
Use the `requirements.txt` to install all the necessary libraries
```bash
pip install -r requirements.txt
```

## 4. Run the Notebooks
```bash
## Install Jupyter
pip install notebook ipykernel

## Register venv as a jupyter kernel 
python -m ipykernel install --user --name=a3-env --display-name "Python (a3-env)"

## Launch Jupyter Notebook
jupyter notebook
```
In the browser, navigate to ```section_*.ipynb```  to run it.

To deactivate the python env just run ```deactivate``` in the terminal

## Report
The detailed findings, methodology, and results are documented in the coursework report:
`report/A3-CW.pdf`

## Declaration of Use of Autogeneration Tools

I acknowledge the use of the following generative AI tools in this coursework:

1. **ChatGPT** (https://chat.openai.com/)  
Used for code refactoring, generating code snippets and boilerplate templates, and accelerating the debugging process.

2. **Perplexity AI** (https://www.perplexity.ai/)  
Used for general research and fact-checking via the "deep research" feature.

A declaration regarding the use of generative AI tools in writing the report is provided within the report itself.


## License
This project is licensed under the MIT License.