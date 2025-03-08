import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' # TODO
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import numpy as np
import json
import argparse
from modeling_bert import BertForSequenceClassification_GCN_Mean, BertForSequenceClassification_GCN_Only
from transformers import (
    AutoConfig,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    BertForSequenceClassification,
    default_data_collator,
    set_seed,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--train_data', type=str, default='data/zh_baike_qwen-turbo/train_coref.jsonl')
    # parser.add_argument('--valid_data', type=str, default='data/zh_baike_qwen-turbo/test_coref.jsonl')
    parser.add_argument('--output_dir', type=str, default='experiments/zh_baike_qwen-turbo_baseline')
    parser.add_argument('--model', type=str, default="/data/home/models/chinese-roberta-wwm-ext/")
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--seed', type=int, default=66, help='random seed')
    parser.add_argument('--do_train', default=True, help="to train or not to train")
    parser.add_argument('--model_type', type=str, required=True, help="baseline or gcn")
    parser.add_argument('--dataset', type=str, required=True)
    # parser.add_argument('--model_type', type=str, default='baseline', help="baseline or gcn")
    # parser.add_argument('--dataset', type=str, default='baike_qwen_par')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if args.model_type == 'baseline':
        from utils_data_zh import load_data, MyDataset
    else:
        from AIGTDet.script_zh.utils_data_graph import load_data, MyDataset
    if args.dataset != '':
        args.train_data = f'{args.dataset}/train_coref.jsonl'
        args.valid_data = f'{args.dataset}/test_coref.jsonl'
        # args.output_dir = f'experiments_zh/{args.dataset}_{args.model_type}_{args.seed}'
    
    print("args",args)
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))
    
    # Set seed before initializing model, for reproduction purpose.
    set_seed(args.seed)

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.model_type == 'baseline':
        model = BertForSequenceClassification.from_pretrained(args.model, config=config)
        for param in model.parameters(): param.data = param.data.contiguous()
    else:
        # model = BertForSequenceClassification_GCN_Mean.from_pretrained(args.model, config=config)
        model = BertForSequenceClassification_GCN_Only.from_pretrained(args.model, config=config)
        for param in model.parameters(): param.data = param.data.contiguous()
 
    # Load data
    train_data = load_data(args.train_data)
    # label_list = set(train_data[1])
    # label_to_id = {v: i for i, v in enumerate(label_list)}
    label_to_id = {'human': 0, 'machine': 1}
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
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
            }

    training_args = TrainingArguments(
            output_dir = args.output_dir,
            do_train=args.do_train,
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