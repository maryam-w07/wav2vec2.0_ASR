from evaluate import load

wer_metric = load("wer")

def compute_metrics(pred):

    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = load("wer")
    wer_score = wer.compute(predictions=pred_str, references=label_str)

    return {"wer": wer_score}

def map_to_result(batch, model):
  with torch.no_grad():
    input_values = torch.tensor(batch["input_values"], device=device).unsqueeze(0)
    logits = model(input_values).logits

  pred_ids = torch.argmax(logits, dim=-1)
  batch["pred_str"] = processor.batch_decode(pred_ids)[0]
  batch["text"] = processor.decode(batch["labels"], group_tokens=False)

  return batch
