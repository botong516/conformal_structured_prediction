The repo contains the code and results for the submission *Conformal Structured Prediction*

## Directory Structure Overview
- **data/**: Contains scripts for collecting and preprocessing the data files used in the experiments.
  - **squad.py**: Defines the `SQuAD` class, which preprocesses the original dataset and sets up the two-shot prompt.
  - **squad_utils.py**: Provides functions to save and load year-based problems.
  - **mnist.py**: Trains and saves the dataset used for the MNIST experiment.
  - **imagenet_utils.py**: Contains helper functions to read the hierarchy from *WordNet* and identify parent nodes.
- **models/hf_inference.py**: Defines model classes for generating outputs using Hugging Face models.
- **results/**: Contains generated results and plots.
  - **experiments/**: Experimental results data.
  - **plots/**: Experimental results plots. The plots presented in the paper can be reproduced by directly running the Python scripts in this directory.
  - **qual/**: Qualitative example outputs.
- **run/**: Execution files that contain our algorithms for structured conformal prediction for each task. Users can generate results and reproduce the experiments by running these files.
- **structures/**: Defines the DAG structures and the integer programming problem for each task.
- **utils/**: Contains scripts for generating plots.

## Collected Data
All collected data files used in the experiment can be found and downloaded from this [link](https://drive.google.com/file/d/1Vv6BNtydbNHM57NPbE2wr-wS0lbShj0c/view?usp=sharing).

## Steps
1. Download the collected data and place the folder **collected/** in the root directory.
2. Create a file named `hf_token.txt` in the root directory and add your Hugging Face token.
3. Set up the GUROBI optimizer [license](https://www.gurobi.com/solutions/licensing/) if needed.
4. To run the algorithm for each task, navigate to the **run/** directory and execute the corresponding Python script (e.g., `python squad.py`, `python mnist.py`, `python imagenet.py`). Experiment parameters can be modified in the top section of each execution file.
