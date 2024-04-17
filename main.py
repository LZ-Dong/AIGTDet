import os
import numpy as np
import json
import argparse
from modeling_graph import RobertaForSequenceClassification_RGCN, RobertaForSequenceClassification
from utils_data import load_data, MyDataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='OpenLLMText/Human_GPT2/train_coref.jsonl')
    parser.add_argument('--valid_data', type=str, default='OpenLLMText/Human_GPT2/valid_coref.jsonl')
    parser.add_argument('--model', type=str, default='roberta-base')
    parser.add_argument('--output_dir', type=str, default='experiments')
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--seed', type=int, default=66, help='random seed')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print("args",args)
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))
    
    # Set seed before initializing model, for reproduction purpose.
    set_seed(args.seed)

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = RobertaForSequenceClassification_RGCN.from_pretrained(args.model, config=config)
 
    # Load data
    train_data = load_data(args.train_data)
    label_list = set(train_data[1])
    label_to_id = {v: i for i, v in enumerate(label_list)}
    print(label_to_id)
    train_dataset = MyDataset(train_data, tokenizer, args.max_length, False, label_to_id)
    eval_data = load_data(args.valid_data)
    eval_dataset = MyDataset(eval_data, tokenizer, args.max_length, False, label_to_id)

    # def compute_metrics(p: EvalPrediction):
    #     preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    #     preds = np.argmax(preds, axis=1)
    #     correct = ((preds == p.label_ids).sum()).item()
    #     return {'accuracy': 1.0*correct/len(preds)}
    
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        labels = p.label_ids
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
            }

    training_args = TrainingArguments(
            output_dir = args.output_dir,
            do_train=True,
            do_eval=True,
            do_predict=True,
            logging_strategy="steps",
            save_strategy="no",
            learning_rate= args.lr,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.bs,
            num_train_epochs=args.epoch,
            report_to="none"
        )
    
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # if training_args.do_predict:
    #     predictions = trainer.predict(test_dataset, metric_key_prefix="predict").predictions
    #     predictions = np.argmax(predictions, axis=1)
    #     output_predict_file = os.path.join(args.output_dir, "predict_results.txt")
    #     if trainer.is_world_process_zero():
    #         with open(output_predict_file, "w") as writer:
    #             writer.write("index\tprediction\n")
    #             for index, item in enumerate(predictions):
    #                 item = label_list[item]
    #                 writer.write(f"{index}\t{item}\n")

if __name__ == "__main__":
    main()