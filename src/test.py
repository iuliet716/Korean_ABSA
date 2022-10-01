import copy
import os

import jsonlines
import torch

from arguments import get_test_args
from model import RoBertaBaseClassifier
from preprocess import preprocess
from utils import jsonlload

args = get_test_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_from_korean_form(tokenizer, ce_model, pc_model, entity_property_pair, label_id_to_name, polarity_id_to_name, data):
    ce_model.to(device)
    ce_model.eval()
    count = 0
    for sentence in data:
        form = sentence['sentence_form']
        sentence['annotation'] = []
        count += 1
        if type(form) != str:
            print("form type is wrong: ", form)
            continue
        for pair in entity_property_pair:
            tokenized_data = tokenizer(form, pair, padding='max_length', max_length=256, truncation=True)

            input_ids = torch.tensor([tokenized_data['input_ids']]).to(device)
            attention_mask = torch.tensor([tokenized_data['attention_mask']]).to(device)
            with torch.no_grad():
                _, ce_logits = ce_model(input_ids, attention_mask)

            ce_predictions = torch.argmax(ce_logits, dim = -1)

            ce_result = label_id_to_name[ce_predictions[0]]

            if ce_result == 'True':
                with torch.no_grad():
                    _, pc_logits = pc_model(input_ids, attention_mask)

                pc_predictions = torch.argmax(pc_logits, dim=-1)
                pc_result = polarity_id_to_name[pc_predictions[0]]

                sentence['annotation'].append([pair, pc_result])

    return data

if __name__ == "__main__":
    # Create directories for output path
    os.makedirs(args.output_dir, exist_ok=True)


    # Get jsonlines files for test
    print("Get jsonlines files for test")
    test_data = jsonlload(args.test_data)


    # Preprocess test data
    print("Preprocess test data")
    test_dataloader = preprocess(args, test_data, TEST=True)
    entity_property_test_dataloader = test_dataloader.entity_property_dataloader
    polarity_test_dataloader = test_dataloader.polarity_dataloader


    # Get some objects from preprocessing
    entity_property_pair = test_dataloader.entity_property_pair
    label_id_to_name = test_dataloader.label_id_to_name
    polarity_id_to_name = test_dataloader.polarity_id_to_name
    tokenizer = test_dataloader.tokenizer


    # Load model for entity property
    print("Load model for entity property")
    entity_property_model = RoBertaBaseClassifier(args, len(label_id_to_name), len(tokenizer))
    entity_property_model.load_state_dict(torch.load(args.entity_property_model_path, map_location=device))
    entity_property_model.to(device)
    entity_property_model.eval()

    # Load model for polarity
    print("Load model for polarity")
    polarity_model = RoBertaBaseClassifier(args, len(polarity_id_to_name), len(tokenizer))
    polarity_model.load_state_dict(torch.load(args.polarity_model_path, map_location=device))
    polarity_model.to(device)
    polarity_model.eval()


    # Predict result
    print("Predict result")
    pred_data = predict_from_korean_form(tokenizer, entity_property_model, polarity_model, entity_property_pair, label_id_to_name, polarity_id_to_name, copy.deepcopy(test_data))
    with jsonlines.open(args.output_dir + "/predict.jsonl", "w") as fobj:
        for sentence in pred_data:
            fobj.write(sentence)