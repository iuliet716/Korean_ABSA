from collections import namedtuple
import os
from pathlib import Path

import mlflow
from mlflow.pytorch import log_state_dict
import torch
from torch.nn.utils import clip_grad_norm_
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import trange

from arguments import get_train_args
from evaluate import evaluation
from model import RoBertaBaseClassifier
from preprocess import preprocess
from utils import jsonlload


args = get_train_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_configuration(args, model, train_dataloader, FULL_FINETUNING=True):
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.eps
    )
    epochs = args.num_train_epochs
    total_steps = epochs * len(train_dataloader)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    return namedtuple(typename='return_model_configuration',
        field_names=[
            'optimizer',
            'scheduler'
        ]
    )(
        optimizer,
        scheduler
    )


def training(model, train_dataloader, optimizer, scheduler, model_path, epoch_step, max_grad_norm):
    total_loss = 0

    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        model.zero_grad()

        loss, _ = model(b_input_ids, b_input_mask, b_labels)

        loss.backward()

        total_loss += loss.item()

        clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print("Average train loss: {}".format(avg_train_loss))

    # Log average train loss to MLflow
    mlflow.log_metric("Average Train Loss", avg_train_loss)

    model_save_path = model_path + '/epoch_' + str(epoch_step) + '.pt'
    torch.save(model.state_dict(), model_save_path)

    # Add model to the MLflow model registry
    log_state_dict(model.state_dict(), artifact_path=model_path.split("/")[-1] + "_epoch_" + str(epoch_step))

    return namedtuple(typename='return_model_configuration',
        field_names=[
            'model',
            'optimizer',
            'scheduler'
        ]
    )(
        model,
        optimizer,
        scheduler
    )


def evaluating(model, dev_dataloader, label_id_to_name, epoch_step):
    model.eval()

    pred_list = []
    label_list = []

    for batch in dev_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            loss, logits = model(b_input_ids, b_input_mask, b_labels)

        predictions = torch.argmax(logits, dim=-1)
        pred_list.extend(predictions)
        label_list.extend(b_labels)

    evaluation(label_list, pred_list, len(label_id_to_name), epoch_step)

    return model


if __name__ == "__main__":
    # Create directories for model path
    os.makedirs(args.entity_property_model_path, exist_ok=True)
    os.makedirs(args.polarity_model_path, exist_ok=True)

    
    # Create MLflow client
    mlflow_client = mlflow.client.MlflowClient()


    # Create/Get MLflow experiment
    try:
        print("Create MLflow experiment: ", args.experiment_name)
        experiment_id = mlflow_client.create_experiment(
            args.experiment_name,
            artifact_location=Path.cwd().joinpath("mlruns").as_uri()
        )
    except mlflow.exceptions.MlflowException:
        print("The MLflow experiment already exists: ", args.experiment_name)
        experiment_id = mlflow_client.get_experiment_by_name(args.experiment_name).experiment_id


    # Create and Start MLflow run
    mlflow.start_run(
        experiment_id=experiment_id,
        run_name=args.run_name
    )


    # Log hyperparameter to MLflow
    params = {
        "batch size": args.batch_size,
        "learning rate": args.learning_rate,
        "eps": args.eps,
        "train epochs": args.num_train_epochs,
        "base model": args.base_model,
        "max length": args.max_len,
        "classifer hidden size": args.classifier_hidden_size,
        "classifier dropout prob": args.classifier_dropout_prob
    }
    mlflow.log_params(params)


    # Get jsonlines files for train/validation
    print("Get jsonlines files for train/validation")
    train_data = jsonlload(args.train_data)
    dev_data = jsonlload(args.dev_data)


    # Preprocess train data
    print("Preprocess train data")
    train_dataloader = preprocess(args, train_data, TEST=False)
    entity_property_train_dataloader = train_dataloader.entity_property_dataloader
    polarity_train_dataloader = train_dataloader.polarity_dataloader


    # Preprocess validation data
    print("Preprocess validation data")
    dev_dataloader = preprocess(args, dev_data, TEST=False)
    entity_property_dev_dataloader = dev_dataloader.entity_property_dataloader
    polarity_dev_dataloader = dev_dataloader.polarity_dataloader


    # Get some objects from preprocessing
    label_id_to_name = train_dataloader.label_id_to_name
    polarity_id_to_name = train_dataloader.polarity_id_to_name
    tokenizer = train_dataloader.tokenizer


    # Load model for entity property
    print("Load model for entity property")
    entity_property_model = RoBertaBaseClassifier(args, len(label_id_to_name), len(tokenizer))
    entity_property_model.to(device)
    entity_property_model_configuration = model_configuration(args, entity_property_model, entity_property_train_dataloader, True)
    entity_property_optimizer = entity_property_model_configuration.optimizer
    entity_property_scheduler = entity_property_model_configuration.scheduler


    # Load model for polarity
    print("Load model for polarity")
    polarity_model = RoBertaBaseClassifier(args, len(polarity_id_to_name), len(tokenizer))
    polarity_model.to(device)
    polarity_model_configuration = model_configuration(args, polarity_model, polarity_train_dataloader, True)
    polarity_optimizer = polarity_model_configuration.optimizer
    polarity_scheduler = polarity_model_configuration.scheduler


    # Train
    print("Start training")
    epochs = args.num_train_epochs
    epoch_step = 0

    for _ in trange(epochs, desc="Epoch"):
        entity_property_model.train()
        epoch_step += 1
        print("Epoch: ", epoch_step)

        # Entity Property
        print("Entity Property")
        each_train = training(
            entity_property_model,
            entity_property_train_dataloader,
            entity_property_optimizer,
            entity_property_scheduler,
            args.entity_property_model_path,
            epoch_step,
            max_grad_norm=1.0           
        )
        entity_property_model = each_train.model
        entity_property_optimizer = each_train.optimizer
        entity_property_scheduler = each_train.scheduler

        if args.do_eval:
            each_evaluate = evaluating(
                entity_property_model,
                entity_property_dev_dataloader,
                label_id_to_name,
                epoch_step              
            )
            entity_property_model = each_evaluate


        # Polarity
        print("Polarity")
        each_train = training(
            polarity_model,
            polarity_train_dataloader,
            polarity_optimizer,
            polarity_scheduler,
            args.polarity_model_path,
            epoch_step,
            max_grad_norm=1.0          
        )
        polarity_model = each_train.model
        polarity_optimizer = each_train.optimizer
        polarity_scheduler = each_train.scheduler

        if args.do_eval:
            each_evaluate = evaluating(
                polarity_model,
                polarity_dev_dataloader,
                polarity_id_to_name,
                epoch_step
            )
            polarity_model = each_evaluate

mlflow.end_run()