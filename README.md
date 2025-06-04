# # ChatGPT Small Fine-tuning (Minimal Version Without Validation)

## Project Goals

This project fine-tunes a small GPT model (`ChatGPT Small Rus 3 / ruGPT3Small`) on Ukrainian-language banking data from Terms & Conditions of PrivatBank to:

- Generate correct and concise banking product descriptions in Ukrainian.
- Respond to banking-related customer queries.
- Be used in chatbots, FAQ systems, or digital assistant tools.

---

## How to Run in Google Colab (Legacy Transformers-Compatible)

### Prerequisites:
- Python ≥ 3.8
- GPU-enabled Google Colab
- Transformers < 4.25 supported

### Install dependencies:
```bash
pip install transformers datasets peft accelerate evaluate sacrebleu rouge_score --quiet
```

### Upload the dataset:
```python
from google.colab import files
uploaded = files.upload()  # upload privatbank_dataset_ua.jsonl
```

### Prepare the dataset:
```python
import json
from datasets import Dataset

data = [json.loads(line) for line in open("privatbank_dataset_ua.jsonl", encoding="utf-8")]
dataset = Dataset.from_list(data)
```

### Load the model:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "sberbank-ai/rugpt3small_based_on_gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
```

### Tokenization:
```python
def tokenize(example):
    return tokenizer(f"<s>Запит: {example['input']}\nВідповідь: {example['output']}</s>", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize)
```

### Train without validation:
```python
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

training_args = TrainingArguments(
    output_dir="./finetuned-chatgpt-banking",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_steps=50,
    save_steps=100,
    save_total_limit=2,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()
```

### Save the model:
```python
trainer.save_model("./finetuned-chatgpt-banking")
tokenizer.save_pretrained("./finetuned-chatgpt-banking")
```

---

## Example Request
```python
prompt = "Що таке кредит під депозит?"

inputs = tokenizer(f"<s>Запит: {prompt}\nВідповідь:", return_tensors="pt")
outputs = model.generate(**inputs, max_length=150)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Notes

- This version skips validation and does not use evaluation_strategy, save_strategy, or best model tracking.
- Compatible with older `transformers` versions that lack modern Trainer features.
