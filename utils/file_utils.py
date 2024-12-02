import datetime
import yaml
import os


class MainFileManage:
    def __init__(self, config_name, name, base_config, log_dir='logs/'):
        self.now = datetime.datetime.now().strftime("D%Y-%m-%dT%H-%M-%S")
        self.config_name = config_name
        self.name = name
        self.log_dir = log_dir

        self.save_log_name = self.now + '_' + self.name
        self.save_log_path = os.path.join(self.log_dir, self.config_name, self.save_log_name)

        self.save_tensorboard_log_path = os.path.join(self.save_log_path, 'tensorboard')
        self.save_checkpoints_path = os.path.join(self.save_log_path, 'checkpoints')
        self.save_config_path = os.path.join(self.save_log_path, 'configs')

        self.project_config_name = self.now + '-project.yaml'
        self.lighting_config_name = self.now + '-lighting.yaml'

        self.base_config = base_config

        self.create_log_dir()
        self.save_model_config()

    def create_log_dir(self):
        os.makedirs(self.save_log_path, exist_ok=True)
        os.makedirs(self.save_tensorboard_log_path, exist_ok=True)
        os.makedirs(self.save_checkpoints_path, exist_ok=True)
        os.makedirs(self.save_config_path, exist_ok=True)

    def save_model_config(self):
        base_config = self.base_config
        project_config = {
            'model': base_config['model'],
            'data': base_config['data'],
        }
        lighting_config = {
            'lighting' : base_config['lightning']
        }

        with open(os.path.join(self.save_config_path, self.project_config_name), 'w') as file:
            yaml.dump(project_config, file, default_flow_style=False, sort_keys=False)

        with open(os.path.join(self.save_config_path, self.lighting_config_name), 'w') as file:
            yaml.dump(lighting_config, file, default_flow_style=False, sort_keys=False)

