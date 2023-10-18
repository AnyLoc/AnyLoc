
import sys
import torch
import logging
import multiprocessing
from datetime import datetime

import test
import parser
import commons
from model import network
from datasets.test_dataset import TestDataset

import os
import sys
from pathlib import Path
# Set the './../' from the script folder
dir_name = None
try:
    dir_name = os.path.dirname(os.path.realpath(__file__))
except NameError:
    print('WARN: __file__ not found, trying local')
    dir_name = os.path.abspath('')
lib_path = os.path.realpath(f'{Path(dir_name).parent}')
# Add to path
if lib_path not in sys.path:
    print(f'Adding library path: {lib_path} to PYTHONPATH')
    sys.path.append(lib_path)
else:
    print(f'Library path {lib_path} already in PYTHONPATH')

from dvgl_benchmark.datasets_ws import BaseDataset
from custom_datasets.baidu_dataloader import Baidu_Dataset
from custom_datasets.oxford_dataloader import Oxford
from custom_datasets.gardens import Gardens
from custom_datasets.aerial_dataloader import Aerial
from custom_datasets.hawkins_dataloader import Hawkins
from custom_datasets.vpair_dataloader import VPAir
from custom_datasets.laurel_dataloader import Laurel
from custom_datasets.eiffel_dataloader import Eiffel
from custom_datasets.vpair_distractor_dataloader import VPAir_Distractor
from configs import base_dataset_args
import wandb
import time

torch.backends.cudnn.benchmark = True  # Provides a speedup

args = parser.parse_arguments(is_training=False)

#WandB init
if args.use_wandb:
    # Launch WandB
    wandb.init(project=args.wandb_proj, 
            entity=args.wandb_entity, config=args, 
            group=args.wandb_group,name=args.wandb_name)

start_time = datetime.now()
output_folder = f"logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
commons.make_deterministic(args.seed)
commons.setup_logging(output_folder, console="info")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {output_folder}")

#### Model
if args.backbone == "ViT":
    model = network.vit_geo_localization_net()
    args.fc_output_dim = 768    # TODO: Supports only ViT-B/16
else:
    model = network.GeoLocalizationNet(args.backbone, args.fc_output_dim)

logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

if args.resume_model is not None:
    logging.info(f"Loading model from {args.resume_model}")
    model_state_dict = torch.load(args.resume_model)
    res = model.load_state_dict(model_state_dict)
    logging.info(f"Model state dict loaded: {res}")
else:
    logging.warning("WARNING: You didn't provide a path to resume the model (--resume_model parameter). " +
                 "Evaluation will be computed using randomly initialized weights.")

model = model.to(args.device)

if 'sf' in args.dataset_folder:
    queries_folder="queries_v1"
else:
    queries_folder="queries"

if args.dataset_name=='baidu_datasets':
    test_ds = Baidu_Dataset(args,args.dataset_folder,args.dataset_name)
elif args.dataset_name=="Oxford":
    test_ds = Oxford(args.dataset_folder)
elif args.dataset_name=="Oxford_25m":
    test_ds = Oxford(args.dataset_folder, override_dist=25)
elif args.dataset_name=="gardens":
    test_ds = Gardens(args,args.dataset_folder,args.dataset_name)
elif args.dataset_name.startswith("hawkins"):
    test_ds = Hawkins(args,args.dataset_folder,"hawkins_long_corridor")
elif args.dataset_name=="VPAir":
    test_ds = VPAir(args,args.dataset_folder,args.dataset_name)
    test_distractor_ds = VPAir_Distractor(args,args.dataset_folder,args.dataset_name)
elif args.dataset_name=="laurel_caverns":
    test_ds = Laurel(args,args.dataset_folder,args.dataset_name)
elif args.dataset_name.startswith("Tartan_GNSS"):
    test_ds = Aerial(args,args.dataset_folder,args.dataset_name)
elif args.dataset_name=="eiffel":
    test_ds = Eiffel(args,args.dataset_folder,args.dataset_name)
else:
    # test_ds = TestDataset(args.test_set_folder, queries_folder=queries_folder,
    #                 positive_dist_threshold=args.positive_dist_threshold,resize=args.resize)
    test_ds = BaseDataset(base_dataset_args, args.dataset_folder, args.dataset_name, "test")

if args.dataset_name=="VPAir":
    recalls, recalls_str = test.test(args, test_ds, model, test_distractor_ds)
else:
    recalls, recalls_str = test.test(args, test_ds, model)
logging.info(f"{test_ds}: {recalls_str}")

results = {}
recalls_val = args.recall_values or [1,5,10,20]
for tk in range(recalls.shape[0]):
    results[f"R@{recalls_val[tk]}"] = recalls[tk]/100
results["Agg-Method"] = "Global"
ts = time.strftime(f"%Y_%m_%d_%H_%M_%S")
results["Timestamp"] = str(ts)
results["DB-Name"] = str(args.dataset_name)

if args.use_wandb:
    wandb.log(results)
    
    # Log to Wandb
    for tk in range(recalls.shape[0]):
        wandb.log({"Recall-All": recalls[tk]}, step=int(tk))
    # Close Wandb
    wandb.finish()
else:
    logging.info("Not logging results to WandB")
    logging.info(results)
