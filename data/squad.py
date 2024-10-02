import datasets
import functools

@functools.lru_cache(1)
class SQuAD():
    def __init__(self, split='validation'):
        self.split = split
        self.dataset = datasets.load_dataset('squad', split=split)
    
    def get_dataset(self, tokenizer=None, prompt_template=None):

        def verbalize(example):
            if prompt_template is not None:
                example['prompt'] = prompt_template(example)
            else:
                question = example['question']
                context = example['context']
                example['prompt'] = f"""You are an expert in answering knowledge-intensive questions. The following question is about years. Provide the answer as a four-digit year like '2015'. Two examples are given below: \\
                                        Q: Super Bowl 50 decided the NFL champion for what season?
                                        Context: Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50. \\
                                        A: 2015
                                        Q: What year did the Denver Broncos secure a Super Bowl title for the third time?
                                        Context: Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50. \\
                                        A: 2016
                                        Q: {question}
                                        Context: {context}
                                        A: """           
                example['answers'] = example['answers']['text']
                if tokenizer is not None:
                    inputs = tokenizer(example['prompt'], padding=False, truncation=False)
                    outputs = tokenizer(example['answers'][0], padding=False, truncation=False)
                    example['input_ids'] = inputs['input_ids']
                    example["attention_mask"] = inputs.attention_mask
                    example["labels"] = outputs.input_ids.copy()
                    example["labels"] = [-100 if _ == tokenizer.pad_token_id else _ for _ in example["labels"]]
            return example

        self.dataset = self.dataset.map(verbalize, load_from_cache_file=False)
        if tokenizer is not None:
            self.dataset.set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "labels"],
                output_all_columns=True)
        return self.dataset