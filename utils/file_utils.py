import datetime
import yaml
import os


class MainFileManage:
    def __init__(self, config_name, name, base_config, resume_model=None, log_dir='logs/'):
        if resume_model is None:
            now = datetime.datetime.now().strftime("D%Y-%m-%dT%H-%M-%S")
            self.save_log_name = now + '_' + name
            self.save_log_path = os.path.join(log_dir, config_name, self.save_log_name)
            self.save_tensorboard_log_path = os.path.join(self.save_log_path, 'tensorboard')
            self.save_checkpoints_path = os.path.join(self.save_log_path, 'checkpoints')
            self.save_config_path = os.path.join(self.save_log_path, 'configs')

            self.project_config_name = now + '-project.yaml'
            self.lighting_config_name = now + '-lighting.yaml'

            self.base_config = base_config
            self.create_log_dir()
            self.save_model_config()
        else:
            self.save_checkpoints_path, _ = os.path.split(resume_model)
            self.save_log_path, _ = os.path.split(self.save_checkpoints_path)
            _, self.save_log_name = os.path.split(self.save_log_path)
            self.save_tensorboard_log_path = os.path.join(self.save_log_path, 'tensorboard')
            self.save_config_path = os.path.join(self.save_log_path, 'configs')

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

