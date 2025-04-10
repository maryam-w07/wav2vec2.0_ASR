from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./local_model",
    group_by_length=False,           # For IterableDataset, grouping is disabled.
    max_steps=3000,                  # Train for 1000 steps.
    per_device_train_batch_size=4,   
    evaluation_strategy="steps",
    num_train_epochs=1,              .
    fp16=True,                       
    gradient_checkpointing=False,    .
    save_steps=250,                  # Save checkpoints every 250 steps.
    eval_steps=250,                  # Evaluate every 250 steps.
    logging_steps=50,                # Log training metrics every 50 steps.
    learning_rate=1e-4,
    weight_decay=0.005,
    warmup_steps=100,                # A higher warmup helps with training stability.
    save_total_limit=1,
    push_to_hub=False,
    disable_tqdm=False,              
    report_to="none"
)
from transformers import Trainer

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=processor.tokenizer,
)
trainer.train()
