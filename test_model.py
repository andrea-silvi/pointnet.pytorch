from utils.dataset import *
from train_ae import upload_args_from_json, test_example
from pointnet.pointnet_model import *
from pointnet.deeper_pointnet_model import *
from gcnn.gcnn_model import *
import torch.utils.data
from utils.loss import *
import neptune.new as neptune


def evaluate_loss_by_class(opt, autoencoder, run):
    run["params"] = vars(opt)
    classes = ["Airplane", "Car", "Chair", "Lamp", "Mug", "Motorbike", "Table"] if opt.test_class_choice=="None" \
        else [opt.test_class_choice]
    autoencoder.cuda()
    for classs in classes:
        test_dataset = ShapeNetDataset(opt.dataset,
                                       opt.num_points,
                                       class_choice=classs,
                                       split='test')
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=opt.batchSize,
            shuffle=True,
            num_workers=int(opt.workers))
        run[f"loss/{classs}"] = test_example(opt, test_dataloader, autoencoder)
