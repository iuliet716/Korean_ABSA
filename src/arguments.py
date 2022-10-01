import argparse

def get_train_args():
    parser = argparse.ArgumentParser(description="sentiment analysis - train")

    parser.add_argument(
        "--experiment_name", type=str, default="default",
        help="MLflow Experiment Name"
    )
    parser.add_argument(
        "--run_name", type=str, default="default",
        help="MLflow Run Name"
    )
    parser.add_argument(
        "--train_data", type=str, default="../data/input_data_v1/train.json",
        help="train file"
    )
    parser.add_argument(
        "--dev_data", type=str, default="../data/input_data_v1/dev.json",
        help="dev file"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-5
    )
    parser.add_argument(
        "--eps", type=float, default=1e-8
    )
    parser.add_argument(
        "--do_eval", action="store_true"
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=3
    )
    parser.add_argument(
        "--base_model", type=str, default="xlm-roberta-base"
    )
    parser.add_argument(
        "--entity_property_model_path", type=str, default="./saved_models/default_path/"
    )
    parser.add_argument(
        "--polarity_model_path", type=str, default="./saved_models/default_path/"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output/default_path/"
    )
    parser.add_argument(
        "--max_len", type=int, default=256
    )
    parser.add_argument(
        "--classifier_hidden_size", type=int, default=768
    )
    parser.add_argument(
        "--classifier_dropout_prob", type=int, default=0.1, help="dropout in classifier"
    )
    args = parser.parse_args()
    return args


def get_test_args():
    parser = argparse.ArgumentParser(description="sentiment analysis - train")

    parser.add_argument(
        "--experiment_name", type=str, default="default",
        help="MLflow Experiment Name"
    )
    parser.add_argument(
        "--run_name", type=str, default="default",
        help="MLflow Run Name"
    )
    parser.add_argument(
        "--test_data", type=str, default="../data/input_data_v1/test.json",
        help="test file"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-5
    )
    parser.add_argument(
        "--eps", type=float, default=1e-8
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=3
    )
    parser.add_argument(
        "--base_model", type=str, default="xlm-roberta-base"
    )
    parser.add_argument(
        "--entity_property_model_path", type=str, default="./saved_models/default_path/"
    )
    parser.add_argument(
        "--polarity_model_path", type=str, default="./saved_models/default_path/"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output/default_path/"
    )
    parser.add_argument(
        "--max_len", type=int, default=256
    )
    parser.add_argument(
        "--classifier_hidden_size", type=int, default=768
    )
    parser.add_argument(
        "--classifier_dropout_prob", type=int, default=0.1, help="dropout in classifier"
    )
    args = parser.parse_args()
    return args
