import os
import sys
import json
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data import squad
from models import hf_inference

DATAFILE = '../collected/squad/squad_year_questions_n691.json'

def load_data():
    if os.path.exists(DATAFILE):
        with open(DATAFILE, 'r') as f:
            example_dicts = json.load(f)
        return example_dicts
    
    else:
        dataset = squad.SQuAD()
        count = 0
        example_dicts = []
        for example in dataset.get_dataset():
            raw_answer = example['answers']
            if not isinstance(raw_answer, list):
                raw_answer = [raw_answer]

            answer_set = set()
            for ans in raw_answer:
                if ans.isdigit() and len(ans) == 4: # ensure answer set contains only four-digit numbers
                    answer_set.add(int(ans))
            answer = list(answer_set)

            if len(answer) == 0:
                continue
            else:
                count += 1
                if count == 1 or count == 2: # skip the two questions used in prompt
                    continue
                example_dict = {}
                example_dict['question'] = example['question']
                example_dict['context'] = example['context']
                example_dict['prompt'] = example['prompt']
                example_dict['raw_answer'] = raw_answer
                example_dict['answer'] = answer
                example_dicts.append(example_dict)

        with open(DATAFILE, 'w') as f:
            json.dump(example_dicts, f, indent=4)
        print(f"Saved {len(example_dicts)} SQuAD year problems to {DATAFILE}.")
        return example_dicts



def plot_year_distribution(example_dicts):
    years = []
    for example_dict in example_dicts:
        for answer in example_dict['answer']:
            if 1020 < answer <= 2020:
                years.append(int(answer))
    
    counts, bins, patches = plt.hist(years, bins=range(1020, 2021, 50))
    for count, patch in zip(counts, patches):
        plt.text(patch.get_x() + patch.get_width() / 2, count, str(int(count)), ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.title(f'Distribution of {len(years)} Years (50-Year Bins: 1020–2020)')
    plt.xticks(bins, [f'{int(b)}' for b in bins], rotation=90, ha="right", fontsize=6)
    plt.savefig('squad_year_distribution.jpg')

if __name__ == '__main__':
    # Sanity check
    example_dicts = load_data()
    plot_year_distribution(example_dicts)
