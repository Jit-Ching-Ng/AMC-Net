import os
import shutil

def update_learning_rate(optimizer, lr_scheduler):
    old_lr = optimizer.param_groups[0]['lr']
    lr_scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print('learning rate %.7f -> %.7f' % (old_lr, lr))

def prepare_work(config):
    if not os.path.exists(config.result_dir):
        os.mkdir(config.result_dir)
    experiment_dir = os.path.join(config.result_dir, config.experiment_name)
    if os.path.exists(experiment_dir):
        shutil.rmtree(experiment_dir)
    os.mkdir(experiment_dir)
    phase_dir = os.path.join(experiment_dir, "Train")
    os.mkdir(phase_dir)
    print_options(config)
    return phase_dir


def print_options(config):
    message = ''
    message += '----------------- Options ---------------\n'
    values = vars(config)
    for k, v in sorted(values.items()):
        message += '{:>25}: {:<30}\n'.format(str(k), str(v))
    message += '----------------- End -------------------'
    save_suffix = os.path.join(config.result_dir, config.experiment_name)
    with open(os.path.join(save_suffix, "setting.txt"), "w") as file:
        file.write(message)
    print(message)
