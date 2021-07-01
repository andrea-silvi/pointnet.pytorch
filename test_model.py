from utils.dataset import *
from train_ae import upload_args_from_json, test_example
from pointnet.pointnet_model import *
from pointnet.deeper_pointnet_model import *
from gcnn.gcnn_model import *
import torch.utils.data
from utils.loss import *
import neptune.new as neptune


def evaluate_loss_by_class(model=None):
    opt = upload_args_from_json(os.path.join("parameters", "fixed_params.json"))
    neptune_info = json.loads(open(os.path.join("parameters", "neptune_params.json")).read())
    run = neptune.init(project=neptune_info['project'],
                       tags=["TEST LOSS BY CLASS", opt.type_encoder, opt.architecture, opt.type_decoder],
                       api_token=neptune_info['api_token'],)
    run["params"] = vars(opt)

    classes = ["Airplane", "Car", "Chair", "Lamp", "Mug", "Motorbike", "Table"]
    if opt.type_encoder == "pointnet":
        autoencoder = PointNet_DeeperAutoEncoder(opt.num_points, opt.size_encoder, dropout=opt.dropout) \
            if opt.architecture == "deep" else \
            PointNet_AutoEncoder(opt, opt.num_points, opt.size_encoder, dropout=opt.dropout)
    elif opt.type_encoder == 'dgcnn':
        autoencoder = DGCNN_AutoEncoder(opt)
    else:
        raise IOError(f"Invalid type_encoder!! Should be 'pointnet' or 'dgcnn'. Found: {opt.type_encoder}")
    if opt.model == "" and model==None:
        raise IOError("The key 'model' must be defined! Model should be the path of the previous trained model")
    else:
        model = model if opt.model=="" else opt.model
    autoencoder.load_state_dict(torch.load(model))
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

    run.stop()


if __name__=="__main__":
    evaluate_loss_by_class()