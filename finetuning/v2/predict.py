import os

from utils import config, reset_config
from main import main as run


def export():
    out_path = config.get_path('output')

    if not os.path.exists(out_path):
        print('output dir does not exist. skipping')
        return

    checkpoints = sorted([int(d[-3:]) for d in os.listdir(out_path) if d.startswith('checkpoint-')])

    for ckpt in checkpoints:
        test_path = os.path.join(out_path, str(ckpt).zfill(3) + '-test.tsv')
        if os.path.exists(test_path):
            print('skipping, export already exists')
            continue

        config.add('model.checkpoint', ckpt)
        run()


def main():
    data_cfgs = config.data.cfgs if config.data.name is None else [None]
    model_cfgs = config.model.cfgs if config.model.name is None else [None]

    for data_cfg in data_cfgs:
        for model_cfg in model_cfgs:
            print('\nExporting predictions for {}'.format(model_cfg))
            reset_config()
            config.add('util/export')
            if model_cfg is not None:
                config.add(model_cfg)
            if data_cfg is not None:
                config.add(data_cfg)
            export()


if __name__ == '__main__':
    main()
