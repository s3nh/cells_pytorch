import utils as ut
import transfer as tr

import argparse

def build_args():

    parser=argparse.ArgumentParser()
    parser.add_argument('-dir', help = 'path', type = str)
    parser.add_argument('-device', help = 'device', type = str)
    parser.add_argument('-savepath', help = 'savepath', type = str)
    parser.add_argument('-num_epochs', help = 'number of epochs', type = int)
    parser.add_argument('-batch_size', help = 'batch size', type = int)
    args = parser.parse_args()
    return args

def model_save(savepath, model):

    os.makedirs(savepath, exist_ok=True)
    try:
        torch.save(model.state_dict, os.path.join(savepath, model))
    except:
        ValueError("Path does not specified correctly")

    print("Model saved in \n {}".format(os.path.join(savepath, model)))


def main():
    args = build_args()
    image_datasets, dataloaders, dataset_sizes, class_names, device = ut.data_transforms(args.dir, args.device,args.batch_size)
    model_ft, criterion, optimizer_ft, exp_lr_scheduler = tr.finetune_convnet(device)
    model_trained_ = tr.train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, device, dataset_sizes,   num_epochs=args.num_epochs)    
     
    model_save(args.savepath, model_trained_)


if __name__ == "__main__":

    main()
    

