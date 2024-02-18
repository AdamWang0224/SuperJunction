import os
import yaml
import logging
import torch
import numpy as np

from Test_model import Test_model



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
    workers_test = training_params.get('workers_test', 1)  # 16
    logging.info(f"workers_test: {workers_test}")

    data_transforms = {'test': transforms.Compose([transforms.ToTensor()])}

    test_loader = None
    if dataset == 'retina_test':
        from datasets.retina_test_dataset import RetinaDataset
        test_set = RetinaDataset(transform=data_transforms['test'], **config['data'])
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True,
                                                  num_workers=workers_test, worker_init_fn=worker_init_fn)

    return {'test_set': test_set, 'test_loader': test_loader}


def test(config, output_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("test on device: %s", device)

    # data loading
    task = config["data"]["dataset"]
    data = dataLoader(config, dataset=task)
    test_set, test_loader = data["test_set"], data["test_loader"]

    test_agent = Test_model(config["model"], device=device)

    # feed the data into the agent
    # test_agent.test_loader = test_loader
    test_agent.set_test_loader(test_loader)

    # model loading
    test_agent.loadModel()

    # inference to output keypoints
    test_agent.test(output_dir)




if __name__ == "__main__":
    config_path = './configs/magicpoint_retina_repeatability_heatmap.yaml'
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        raise FileNotFoundError("Config File doesn't Exist")

    output_dir = 'logs/superpoint_retina/'
    with open(os.path.join(output_dir, "config_test.yml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # output_dir = output_dir + 'predictions'
    # os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.DEBUG)

    test(config, output_dir)
