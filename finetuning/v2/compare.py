import os
import json

from utils import reset_config, config


def main():
    data_cfgs = config.data.cfgs if config.data.name is None else [None]
    model_cfgs = config.model.cfgs if config.model.name is None else [None]

    for data_cfg in data_cfgs:
        print('')
        for model_cfg in model_cfgs:
            reset_config()
            config.add('util/export', silent=True)
            if model_cfg is not None:
                config.add(model_cfg, silent=True)
            if data_cfg is not None:
                config.add(data_cfg, silent=True)

            out_path = config.get_path('output')
            stats_path = os.path.join(out_path, 'model', 'stats.json')
            if not os.path.exists(stats_path):
                print(f'{stats_path} does not exist. skipping')
                continue

            with open(stats_path) as f:
                stats = json.load(f)
            dev = round(stats["dev_score"], 3)
            test = round(stats["test_score"], 3)
            print(f'{data_cfg:<20}\t{model_cfg}\t{stats["best_checkpoint"]}\t{dev}\t{test}')


if __name__ == '__main__':
    main()
