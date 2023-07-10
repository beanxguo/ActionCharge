import copy


def _read_config(path):
    lines = open(path).readlines()

    runs = []
    run = [1, dict()]
    for line in lines:
        stripped_line = line.strip()

        # continue in case of comment
        if stripped_line.startswith('#'):
            continue

        if not stripped_line:
            if run[1]:
                runs.append(run)

            run = [1, dict()]
            continue

        if stripped_line.startswith('[') and stripped_line.endswith(']'):
            repeat = int(stripped_line[1:-1])
            run[0] = repeat
        else:
            key, value = stripped_line.split('=')
            key, value = (key.strip(), value.strip())
            run[1][key] = value

    if run[1]:
        runs.append(run)

    return runs
def _convert_config(config):
    config_list = []
    for k, v in config.items():
        if v == "None":
            continue
        if v.startswith("["):
            v = v[1:-1].replace(",", "")
        if v.lower() == 'true':
            config_list.append('--' + k)
        elif v.lower() != 'false':
            config_list.extend(['--' + k] + v.split(' '))
    return config_list
def yield_configs(arg_parser, args, verbose=True):
    if args.config:
        config = _read_config(args.config)
    for run_repeat, run_config in config:
        args_copy = copy.deepcopy(args)
        run_config = copy.deepcopy(run_config)
        config_list = _convert_config(run_config)
        run_args = arg_parser.parse_args(config_list,namespace=args_copy)
    return copy.deepcopy(run_args)
