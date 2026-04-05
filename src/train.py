import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

print("1. Loading your supplement data...")

# Load the CSV
df = pd.read_csv("dataset.csv")

# Convert pandas dataframe to Hugging Face Dataset format
hf_dataset = Dataset.from_pandas(df)
# Split the data: 80% for training, 20% for testing the AI's accuracy
split_dataset = hf_dataset.train_test_split(test_size=0.2)

print("2. Tokenizing the ingredients...")
# Load the DistilBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    # Padding and truncation ensure all text is the same length for the AI
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# Apply tokenization to the dataset
tokenized_datasets = split_dataset.map(tokenize_function, batched=True)

print("3. Setting up the DistilBERT Model...")
# Load the pre-trained model and tell it we only have 2 labels (0 and 1)
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

print("4. Configuring Training Parameters...")
training_args = TrainingArguments(
    output_dir="./c0_model_checkpoints", # Where to save progress
    eval_strategy="epoch",               # Test accuracy after every round
    learning_rate=2e-5,                  # How fast the AI learns (keep this small!)
    per_device_train_batch_size=8,       # How many labels to study at once
    num_train_epochs=5,                  # How many times to read the "textbook"
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

print("5. Commencing Training! This may take a few minutes...")
trainer.train()

print("6. Training Complete. Saving the C0 Brain...")
# Save the final trained model so you can use it in your app later!
trainer.save_model("./c0_final_model")
tokenizer.save_pretrained("./c0_final_model")
print("Done! Your custom model is saved in the /c0_final_model folder.")