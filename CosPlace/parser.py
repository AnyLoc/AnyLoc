
import os
import argparse


def parse_arguments(is_training: bool = True):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # CosPlace Groups parameters
    parser.add_argument("--M", type=int, default=10, help="_")
    parser.add_argument("--alpha", type=int, default=30, help="_")
    parser.add_argument("--N", type=int, default=5, help="_")
    parser.add_argument("--L", type=int, default=2, help="_")
    parser.add_argument("--groups_num", type=int, default=8, help="_")
    parser.add_argument("--min_images_per_class", type=int, default=10, help="_")
    # Model parameters
    parser.add_argument("--backbone", type=str, default="ResNet101",
                        choices=["VGG16", "ResNet18", "ResNet50", "ResNet101", "ResNet152", "ViT"], help="_")
    parser.add_argument("--fc_output_dim", type=int, default=2048,
                        help="Output dimension of final fully connected layer")
    # Training parameters
    parser.add_argument("--use_amp16", action="store_true",
                        help="use Automatic Mixed Precision")
    parser.add_argument("--augmentation_device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="on which device to run data augmentation")
    parser.add_argument("--batch_size", type=int, default=32, help="_")
    parser.add_argument("--epochs_num", type=int, default=50, help="_")
    parser.add_argument("--iterations_per_epoch", type=int, default=10000, help="_")
    parser.add_argument("--lr", type=float, default=0.00001, help="_")
    parser.add_argument("--classifiers_lr", type=float, default=0.01, help="_")
    # Data augmentation
    parser.add_argument("--brightness", type=float, default=0.7, help="_")
    parser.add_argument("--contrast", type=float, default=0.7, help="_")
    parser.add_argument("--hue", type=float, default=0.5, help="_")
    parser.add_argument("--saturation", type=float, default=0.7, help="_")
    parser.add_argument("--random_resized_crop", type=float, default=0.5, help="_")
    # Validation / test parameters
    parser.add_argument("--infer_batch_size", type=int, default=16,
                        help="Batch size for inference (validating and testing)")
    parser.add_argument("--positive_dist_threshold", type=int, default=25,
                        help="distance in meters for a prediction to be considered a positive")
    parser.add_argument('--resize', type=int, default=[480, 640], nargs=2, help="Resizing shape for images (HxW).")
    # Only adding for Baidu dataset
    parser.add_argument('--test_method', type=str, default="hard_resize",
                        choices=["hard_resize", "single_query", "central_crop", "five_crops", "nearest_crop", "maj_voting"],
                        help="This includes pre/post-processing methods and prediction refinement")
    # Resume parameters
    parser.add_argument("--resume_train", type=str, default=None,
                        help="path to checkpoint to resume, e.g. logs/.../last_checkpoint.pth")
    parser.add_argument("--resume_model", type=str, default="/ocean/projects/cis220039p/shared/datasets/vpr/models/cosplace/resnet101_2048.pth",
                        help="path to model to resume, e.g. logs/.../best_model.pth")
    # Other parameters
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="_")
    parser.add_argument("--seed", type=int, default=0, help="_")
    parser.add_argument("--num_workers", type=int, default=8, help="_")
    # Paths parameters
    parser.add_argument("--dataset_name", type=str,default="hawkins_long_corridor", help="dataset name")
    parser.add_argument("--dataset_folder", type=str, default="/home/jay/Downloads/vl_vpr_datasets",
                        help="path of the folder with train/val/test sets")
    parser.add_argument("--save_dir", type=str, default="default",
                        help="name of directory on which to save the logs, under logs/save_dir")
    parser.add_argument('--recall_values', type=int, default=[1, 5, 10, 20], nargs="+",
                        help="Recalls to be computed, such as R@5.")
    parser.add_argument("--save_descs", type=str, default=None,
            help="If not None, then save all descriptors as a numpy file. "\
                "Don't pass the '.npy' extension.")
    #Wandb parameters
    parser.add_argument("--use_wandb",action='store_true',help="Set true to enable wandb logging")
    parser.add_argument("--wandb_entity",type=str,default="vpr-vl",help="keep vpr-vl by default")
    parser.add_argument("--wandb_proj",type=str,default="demo",help="Wandb Project")
    parser.add_argument("--wandb_group",type=str,default="gardens",help="keep vpr-vl by default")
    parser.add_argument("--wandb_name",type=str,default="gardens/CosPlace",help="keep vpr-vl by default")
    parser.add_argument("--wandb_save_qual",type=bool,default=False,help="save qualitiative results")
    
    args = parser.parse_args()

    dataset_list = ['baidu_datasets','Oxford','gardens','hawkins','hawkins_long_corridor','laurel_caverns','VPAir',"Tartan_GNSS_rotated", "Tartan_GNSS_notrotated","Tartan_GNSS_test_rotated","Tartan_GNSS_test_notrotated",'eiffel']
    if args.dataset_name in dataset_list:
        return args

    else :
        if args.dataset_folder is None:
            try:
                args.dataset_folder = os.environ['SF_XL_PROCESSED_FOLDER']
            except KeyError:
                raise Exception("You should set parameter --dataset_folder or export " +
                                "the SF_XL_PROCESSED_FOLDER environment variable as such \n" +
                                "export SF_XL_PROCESSED_FOLDER=/path/to/sf_xl/processed")
        
        if not os.path.exists(args.dataset_folder):
            raise FileNotFoundError(f"Folder {args.dataset_folder} does not exist")
        
        if is_training:
            args.train_set_folder = os.path.join(args.dataset_folder, "train")
            if not os.path.exists(args.train_set_folder):
                raise FileNotFoundError(f"Folder {args.train_set_folder} does not exist")
            
            args.val_set_folder = os.path.join(args.dataset_folder, "val")
            if not os.path.exists(args.val_set_folder):
                raise FileNotFoundError(f"Folder {args.val_set_folder} does not exist")

        if args.dataset_name is not None:
            # handle vpr-bench datasets
            if args.dataset_name in ["17places"]:
                if args.dataset_folder is None:
                    args.dataset_folder = os.path.join(args.dataset_folder, args.dataset_name)
                args.test_set_folder = args.dataset_folder
            else:
                if args.dataset_folder is None:
                    args.dataset_folder = os.path.join(args.dataset_folder, args.dataset_name, 'images')
                # args.test_set_folder = os.path.join(args.dataset_folder, "test")
                args.test_set_folder = args.dataset_folder
        if not os.path.exists(args.test_set_folder):
            print(f"ERROR: Folder {args.test_set_folder} does not exist")
            # raise FileNotFoundError(f"Folder {args.test_set_folder} does not exist")
        
        return args
