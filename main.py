import argparse
from args import train_argparser,eval_argparser
from config import yield_configs
from identify.identifier_train import  IdentifierTrainer
from identify import inpute_file
if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument('mode', type=str, help="Mode: 'train' or 'eval'")
    args, _ = arg_parser.parse_known_args()
    if args.mode == 'train':
        arg_parser = train_argparser()
        args, _ = arg_parser.parse_known_args()
        gpu_queue = [0]
        run_args=yield_configs(arg_parser, args)
        device_id = str(gpu_queue[0])
        run_args.device_id=device_id
        trainer = IdentifierTrainer(run_args)
        trainer.train(train_path=run_args.train_path, valid_path=run_args.valid_path,
                      input_reader_cls=inpute_file.JsonInputReader)

    elif args.mode == 'eval':
        arg_parser = eval_argparser
        args, _ = arg_parser.parse_known_args()
        gpu_queue = [0]
        run_args = yield_configs(arg_parser, args)
        device_id = str(gpu_queue[0])
        run_args.device_id = device_id
        evaler = IdentifierTrainer(run_args)
        evaler.eval(dataset_path=run_args.test_path,input_reader_cls=inpute_file.JsonInputReader)

    else:
        raise Exception("Mode not in ['train', 'eval'], e.g. 'python identifier.py train ...'")
