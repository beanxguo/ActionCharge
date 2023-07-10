import argparse

def train_argparser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', type=str,default='./configs/example.conf')
    arg_parser.add_argument('--cpu', action='store_true', default=False,
                            help="If true, train/evaluate on CPU even if a CUDA device is available")
    arg_parser.add_argument('--device_id', type=str, default="0", help="gpu device id")
    arg_parser.add_argument('--seed', type=int, default=-1, help="Seed")

    arg_parser.add_argument('--train_path', type=str, help="Path to train dataset")
    arg_parser.add_argument('--valid_path', type=str, help="Path to validation dataset")
    arg_parser.add_argument('--save_path', type=str, help="Path to directory where model checkpoints are stored")
    arg_parser.add_argument('--accusation_type', type=str, help="Path to validation dataset")
    arg_parser.add_argument('--train_batch_size', type=int, default=2, help="Training batch size")
    arg_parser.add_argument('--epochs', type=int, default=2, help="Number of epochs")
    arg_parser.add_argument('--model_path', type=str, help="Path to directory that contains model checkpoints")
    arg_parser.add_argument('--model_type', type=str, default="identifier", help="Type of model")
    arg_parser.add_argument('--prop_drop', type=float, default=0.1, help="Probability of dropout used in Identifier")
    arg_parser.add_argument('--lstm_drop', type=float, default=0.4)
    arg_parser.add_argument('--lstm_layers', type=int, default=1)
    arg_parser.add_argument('--pool_type', type=str, default="max")
    arg_parser.add_argument('--crf_type_count', type=int, default=6)
    arg_parser.add_argument('--lr', type=float, default=5e-5, help="Learning rate")
    arg_parser.add_argument('--lr_warmup', type=float, default=0.1,
                            help="Proportion of total train iterations to warmup in linear increase/decrease schedule")
    arg_parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay to apply")
    arg_parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Maximum gradient norm")
    arg_parser.add_argument('--accusation_FL_gamma', type=int, default=2)

    arg_parser.add_argument('--eval_batch_size', type=int, default=2, help="Training batch size")


    return arg_parser


def eval_argparser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--device_id', type=str, default="0", help="gpu device id")
    arg_parser.add_argument('--test_path', type=str, help="Path to train dataset")
    arg_parser.add_argument('--accusation_type', type=str, help="Path to validation dataset")
    arg_parser.add_argument('--cpu', action='store_true', default=False,
                            help="If true, train/evaluate on CPU even if a CUDA device is available")
    arg_parser.add_argument('--seed', type=int, default=-1, help="Seed")
    arg_parser.add_argument('--save_path', type=str, help="Path to directory where model checkpoints are stored")
    arg_parser.add_argument('--model_path', type=str, help="Path to directory that contains model checkpoints")
    arg_parser.add_argument('--model_type', type=str, default="identifier", help="Type of model")
    arg_parser.add_argument('--prop_drop', type=float, default=0.1, help="Probability of dropout used in Identifier")
    arg_parser.add_argument('--lstm_drop', type=float, default=0.4)
    arg_parser.add_argument('--lstm_layers', type=int, default=1)
    arg_parser.add_argument('--pool_type', type=str, default="max")
    arg_parser.add_argument('--crf_type_count', type=int, default=6)
    arg_parser.add_argument('--eval_batch_size', type=int, default=2, help="Training batch size")
    return arg_parser