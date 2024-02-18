import os
import numpy as np
import yaml
import torch
import logging
from pathlib import Path
from importlib import import_module
from Train_model import Train_model
from datasets.retina_train import retina_train


def worker_init_fn(worker_id):
   """The function is designed for pytorch multi-process dataloader.
   Note that we use the pytorch random generator to generate a base_seed.
   Please try to be consistent.

   References:
       https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed

   """
   base_seed = torch.IntTensor(1).random_().item()
   np.random.seed(base_seed + worker_id)

def dataLoader(config, dataset):
    import torchvision.transforms as transforms
    training_params = config.get('training', {})
    workers_train = training_params.get('workers_train', 1)  # 16
    workers_val = training_params.get('workers_val', 1)

    logging.info(f"workers_train: {workers_train}, workers_val: {workers_val}")

    data_transforms = {'train': transforms.Compose([transforms.ToTensor()]),
                       'val': transforms.Compose([transforms.ToTensor()])}

    mod = import_module('{}.{}'.format('datasets', dataset))
    Dataset = getattr(mod, dataset)
    print(f"dataset: {dataset}")

    train_set = Dataset(transform=data_transforms['train'], task='train', **config['data'])
    # train_set = retina_train(transform=data_transforms['train'], task='train', **config['data'])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config['model']['batch_size'], shuffle=True,
                                               pin_memory=True, num_workers=workers_train, worker_init_fn=worker_init_fn)

    val_set = Dataset(transform=data_transforms['val'], task='val', **config['data'])
    # val_set = retina_train(transform=data_transforms['val'], task='val', **config['data'])
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=config['model']['eval_batch_size'], shuffle=True,
                                             pin_memory=True, num_workers=workers_val, worker_init_fn=worker_init_fn)

    return {'train_loader': train_loader, 'val_loader': val_loader,
            'train_set': train_set, 'val_set': val_set}


def train(config, output_dir):
    torch.set_default_tensor_type(torch.FloatTensor)
    task = config['data']['dataset']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info('train on device: %s', device)

    with open(os.path.join(output_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    save_path = Path(output_dir) / 'checkpoints'
    os.makedirs(save_path, exist_ok=True)
    logging.info('=> will save everything to {}'.format(save_path))

    # data loading
    data = dataLoader(config, dataset=task)
    train_loader, val_loader = data['train_loader'], data['val_loader']

    logging.info('== %s split size %d in %d batches' % \
                 (train_loader, len(train_loader) * config['model']['batch_size'], len(train_loader)))
    logging.info('== %s split size %d in %d batches' % \
                 (val_loader, len(val_loader) * config['model']['batch_size'], len(val_loader)))

    # instantiate train_agent
    train_agent = Train_model(config, save_path=save_path, device=device)

    # feed the data into the agent
    # train_agent.train_loader = train_loader
    # train_agent.val_loader = val_loader
    train_agent.set_train_loader(train_loader)
    train_agent.set_val_loader(val_loader)

    # load model instantiates the model and load the pretrained model (if any)
    train_agent.loadModel()
    train_agent.dataParallel()

    # train model
    train_agent.train()


if __name__ == '__main__':

    config_path = './configs/superpoint_retina_train_heatmap.yaml'
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        raise FileNotFoundError("Config File doesn't Exist")

    output_dir = 'logs/superpoint_retina'
    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.DEBUG)

    train(config, output_dir)
