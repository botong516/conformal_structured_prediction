import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoImageProcessor, AutoModelForImageClassification

with open('../hf_token.txt', "r") as f:
    hf_key = f.read().strip()

class HFInferenceModel:
    def __init__(self, model_name='meta-llama/Meta-Llama-3-70B-Instruct', device=0, **kwargs):
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            use_fast=True,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        kwargs = {**dict(model=model_name, use_fast=True), **kwargs}
    
    @torch.no_grad()
    def generate(self, prompts, **kwargs):
        prompts = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True).to('cuda')
        outputs = self.model.generate(**prompts, **kwargs)
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences,
            outputs.scores,
            normalize_logits=True)
        return self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True), transition_scores, outputs.scores
    
    def get_log_prob(self, prompt, year):
        # Tokenize all prompts and years
        prompt_tokens = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).input_ids
        year_tokens = self.tokenizer(year, return_tensors='pt', padding=True, truncation=True).input_ids

        # Concatenate prompts and years for each item in the batch
        input_tokens = torch.cat([prompt_tokens, year_tokens], dim=1)

        with torch.no_grad():
            outputs = self.model(input_tokens.to(self.model.device))
            logits = outputs.logits

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Calculate log probabilities for each item in the batch
        total_log_probs = []
        for i in range(len(year)):
            prompt_length = (prompt_tokens[i] != self.tokenizer.pad_token_id).sum()
            year_length = (year_tokens[i] != self.tokenizer.pad_token_id).sum()
            target_log_probs = log_probs[i, prompt_length:prompt_length+year_length-1, :].gather(
                1, year_tokens[i, 1:year_length].to(self.model.device).unsqueeze(-1)
            ).squeeze(-1)
            
            total_log_probs.append(target_log_probs.sum().item())

        return total_log_probs

class HFResNetModel:
    def __init__(self, model_name='jsli96/ResNet-18-1', device=0, **kwargs):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name).to(self.device)

    @torch.no_grad()
    def inference(self, image, **kwargs):
        if image.mode != 'RGB':
            image = image.convert('RGB')

        inputs = self.processor(image, return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits
        predicted_label = logits.argmax(-1).item()
        predictions = torch.softmax(logits, dim=-1)
        return predictions.cpu().detach().numpy()[0], predicted_label