import sys
import yaml


class Config:
    def __init__(self, cfg=None):
        self.cfg = {}
        if cfg is not None:
            self.update(cfg)

    def __getattribute__(self, name):
        cfg = object.__getattribute__(self, 'cfg')
        if name not in cfg:
            return object.__getattribute__(self, name)
        return cfg[name]

    def items(self):
        return object.__getattribute__(self, 'cfg').items()

    def update(self, new_cfg):
        cfg = self.cfg

        for key, val in new_cfg.items():
            if type(val) == dict:
                val = Config(val)
                if key in cfg:
                    cfg[key].update(val)
                    continue
            cfg[key] = val

    def add(self, arg, val=None):
        # Manual item
        if val is not None:
            subkeys = arg.split('.')
            subconfig = self
            for subkey in subkeys[:-1]:
                subconfig = subconfig.cfg[subkey]
            if subkeys[-1] in subconfig.cfg:
                if type(subconfig.cfg[subkeys[-1]]) == int:
                    val = int(val)
                elif type(subconfig.cfg[subkeys[-1]]) == float:
                    val = float(val)
            subconfig.cfg[subkeys[-1]] = val
            print('{} is set to {}'.format(arg, val))
            return

        # Config file shortcut
        if not arg.endswith('.yaml'):
            arg = 'configs/{}.yaml'.format(arg)

        # Config file
        print('importing config from "{}"'.format(arg))
        with open(arg) as f:
            self.update(yaml.load(f, Loader=yaml.Loader))

    def as_dict(self):
        return {key: (val.as_dict() if isinstance(val, Config) else val) for key, val in self.cfg.items()}

    def show(self, depth=0):
        yaml.dump(self.as_dict(), sys.stdout)

    def get_path(self, name):
        return self.data.cfg[name].format(self.data.name, self.model.shortname)


def init_config():
    config = Config()
    config.add('configs/default.yaml')
    for arg in sys.argv[1:]:
        config.add(*arg.split('='))
    return config


def reset_config():
    global config
    config = init_config()


config = init_config()
