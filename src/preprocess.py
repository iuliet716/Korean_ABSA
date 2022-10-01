from collections import namedtuple
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer


entity_property_pair = [
    '제품 전체#일반', '제품 전체#가격', '제품 전체#디자인', '제품 전체#품질', '제품 전체#편의성', '제품 전체#인지도',
    '본품#일반', '본품#디자인', '본품#품질', '본품#편의성', '본품#다양성',
    '패키지/구성품#일반', '패키지/구성품#디자인', '패키지/구성품#품질', '패키지/구성품#편의성', '패키지/구성품#다양성',
    '브랜드#일반', '브랜드#가격', '브랜드#디자인', '브랜드#품질', '브랜드#인지도', 
]

label_id_to_name = ['True', 'False']
label_name_to_id = {label_id_to_name[i]: i for i in range(len(label_id_to_name))}
# label_name_to_id = {
#   'True': 0,
#   'False': 1
# }

polarity_id_to_name = ['positive', 'negative', 'neutral']
polarity_name_to_id = {polarity_id_to_name[i]: i for i in range(len(polarity_id_to_name))}
# polarity_name_to_id = {
#   'positive': 0,
#   'negative': 1,
#   'neutral': 2
# }

def tokenize_and_align_labels(tokenizer, form, annotations, max_len):
    entity_property_data_dict = {
        'input_ids': [],
        'attention_mask': [],
        'label': []
    }
    polarity_data_dict = {
        'input_ids': [],
        'attention_mask': [],
        'label': []
    }

    for pair in entity_property_pair:
        isPairInOpinion = False
        if pd.isna(form):
            break
        tokenized_data = tokenizer(form, pair, padding='max_length', max_length=max_len, truncation=True)
        for annotation in annotations:
            entity_property = annotation[0]
            polarity = annotation[2]

            # # 데이터가 =로 시작하여 수식으로 인정된경우
            # if pd.isna(entity) or pd.isna(property):
            #     continue

            if polarity == '------------':
                continue


            if entity_property == pair:
                # polarity_count += 1
                entity_property_data_dict['input_ids'].append(tokenized_data['input_ids'])
                entity_property_data_dict['attention_mask'].append(tokenized_data['attention_mask'])
                entity_property_data_dict['label'].append(label_name_to_id['True'])

                polarity_data_dict['input_ids'].append(tokenized_data['input_ids'])
                polarity_data_dict['attention_mask'].append(tokenized_data['attention_mask'])
                polarity_data_dict['label'].append(polarity_name_to_id[polarity])

                isPairInOpinion = True
                break

        if isPairInOpinion is False:
            # entity_property_count += 1
            entity_property_data_dict['input_ids'].append(tokenized_data['input_ids'])
            entity_property_data_dict['attention_mask'].append(tokenized_data['attention_mask'])
            entity_property_data_dict['label'].append(label_name_to_id['False'])

    return entity_property_data_dict, polarity_data_dict

def get_dataset(raw_data, tokenizer, max_len):
    input_ids_list = []
    attention_mask_list = []
    token_labels_list = []

    polarity_input_ids_list = []
    polarity_attention_mask_list = []
    polarity_token_labels_list = []

    for utterance in raw_data:
        entity_property_data_dict, polarity_data_dict = tokenize_and_align_labels(tokenizer, utterance['sentence_form'], utterance['annotation'], max_len)
        input_ids_list.extend(entity_property_data_dict['input_ids'])
        attention_mask_list.extend(entity_property_data_dict['attention_mask'])
        token_labels_list.extend(entity_property_data_dict['label'])

        polarity_input_ids_list.extend(polarity_data_dict['input_ids'])
        polarity_attention_mask_list.extend(polarity_data_dict['attention_mask'])
        polarity_token_labels_list.extend(polarity_data_dict['label'])

    return TensorDataset(torch.tensor(input_ids_list), torch.tensor(attention_mask_list),
                         torch.tensor(token_labels_list)), TensorDataset(torch.tensor(polarity_input_ids_list), torch.tensor(polarity_attention_mask_list),
                         torch.tensor(polarity_token_labels_list))


# Tokenize data
def preprocess(args, json_list, TEST):
    special_tokens_dict = {
        'additional_special_tokens': [
            '&name&',
            '&affiliation&',
            '&social-security-num&',
            '&tel-num&',
            '&card-num&',
            '&bank-account&',
            '&num&',
            '&online-account&'
        ]
    }

    # Load pre-trained base model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)


    # Add special tokens
    tokenizer.add_special_tokens(special_tokens_dict)


    # Convert from json list to TensorDataset
    if not TEST:
        entity_property_data, polarity_data \
        = get_dataset(json_list, tokenizer, args.max_len)
    else:
        entity_property_data \
        = get_dataset(json_list, tokenizer, args.max_len)[0]


    # Convert from TensorDataset to DataLoader for entity property
    entity_property_dataloader = DataLoader(entity_property_data, 
                                    shuffle=True, 
                                    batch_size=args.batch_size
                                )

    # Convert from TensorDataset to DataLoader for polarity
    if not TEST:
        polarity_dataloader = DataLoader(polarity_data,
                                shuffle=True,
                                batch_size=args.batch_size
                            )
    else:
        polarity_dataloader = []

    return namedtuple(typename='return_preprocess', 
        field_names=[
            'entity_property_dataloader',
            'polarity_dataloader',
            'entity_property_pair',
            'label_id_to_name',
            'polarity_id_to_name',
            'tokenizer'
        ]
    )(
        entity_property_dataloader,
        polarity_dataloader,
        entity_property_pair,
        label_id_to_name,
        polarity_id_to_name,
        tokenizer
    )
