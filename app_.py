import utils as ut
import transfer as tr

import argparse

def build_args():

    parser=argparse.ArgumentParser()
    parser.add_argument('-dir', help = 'path', type = str)
    parser.add_argument('-device', help = 'device', type = str)
    parser.add_argument('-savepath', help = 'savepath', type = str)
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
    image_datasets, dataloaders, dataset_sizes, class_names, device = ut.data_transforms(args.dir, args.device)
    print(image_datasets) 
    model_ft, criterion, optimizer_ft, exp_lr_scheduler = tr.finetune_convnet(device)
    model_trained_ = tr.train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,dataloaders, device,  num_epochs=25)    
     
    model_save(args.savepath, model_trained_)


if __name__ == "__main__":

    main()
    

