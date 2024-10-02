print('Start program!')

from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, PILToTensor
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from torch.optim import Optimizer
from transformers import get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import StepLR
from torch.optim import AdamW
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from torch.nn import DataParallel
from torchvision.transforms import Compose, Resize, CenterCrop, Lambda, PILToTensor
from PIL import Image

import copy
from accelerate import Accelerator
import scipy as sp
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.Dataset import DatasetHandler
from src.Loss import DPO_Loss, PPO_Loss, KTO_Loss
from src.utils import Metrics 
from src.Models import *
from config.config import Train_Config
from torch.utils.data import DataLoader, Dataset

import warnings
warnings.filterwarnings("ignore") 
###############################################  Device   ###############################################
# If there's a GPU available...
if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
ngpu = torch.cuda.device_count()

###############################################  Optimization type   ###############################################
YOUR_PATH = 'your_base_path'
sys.stdout = open(f'{YOUR_PATH}/models/train_imagenet/CE/logs.out', 'w')
sys.stderr = open(f'{YOUR_PATH}/models/train_imagenet/CE/errors.rr', 'w')

manualSeed = 0
set_seed(manualSeed)
print("Random Seed: ", manualSeed)

save_model = True

method = 'dpo'  # options: 'dpo', 'kto' or 'CE' 
path = f'{YOUR_PATH}/models/train_imagenet/IPO' 
Train_Dataset = 'imagenet100'   
###############################################  Dataset   ###############################################
train_Config = Train_Config() 

if method == 'dpo':
    config = train_Config.get_dpo_config()
elif method == 'kto':
    config = train_Config.get_kto_config()
elif method == 'ppo':
    config = train_Config.get_ppo_config()
elif method == 'CE':
    config = train_Config.get_ce_config()
else:
    raise Exception("Please choose a correct method type.")


# uncomment below for DPO, KTO 
config['lambda_'] = 0.01
config['beta'] = 0.01
config['ipo'] = True



size = 168
preprocessor = Compose([
    Resize(size=size, interpolation=Image.BICUBIC),
    CenterCrop(size=(size, size)),
    Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),  # Ensure image is RGB
    PILToTensor(),
])

# Define the object
num_typographic = 1
obj = type('obj', (object,), {'dataset': f'{Train_Dataset}', 'num_typographic': num_typographic})

# Initialize the DatasetHandler
handler = DatasetHandler(obj, preprocessor, data_split_ratio=0.8, batch_size=config['batch_size'], pretokenize=True) 

# # Create and save the dataset
# handler.create_data(f"{YOUR_PATH}/datasets/typographic_image_classes_{Test_Dataset}.p")

# Load typographic image classes
handler.load_typographic_image_classes(f"{YOUR_PATH}/datasets/typographic_image_classes_{Test_Dataset}.p")

# Prepare prompts (if needed)
prompts = handler.prepare_prompts()

# Split the data
handler.split_data()
# handler.filter_dataset()

if method == 'ppo':
    handler.ppo_process(dataset_name=Train_Dataset)


# Get DataLoaders
train_dataloader = handler.get_train_dataloader()
test_dataloader = handler.get_test_dataloader()
val_dataloader = handler.get_val_dataloader()


# Print to verify the setup
print(f"Number of classes: {handler.NUM_CLASSES}")
print(f"Train dataset size: {len(handler.train_subset)}")
print(f"Validation dataset size: {len(handler.val_subset)}")
print(f"Test dataset size: {len(handler.test_subset)}")

###############################################  Define Model   ###############################################

# Load pre-trained Clip 
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

clip_policy = ClipPolicy(clip_model, clip_processor,
                         handler.NUM_CLASSES,
                         config['batch_size'],
                          device).to(device)

if (method != 'dpo') and (ngpu > 1):
    clip_policy = DataParallel(clip_policy)

ref_policy  = copy.deepcopy(clip_policy)
for param in ref_policy.parameters():
    param.requires_grad = False
clip_policy = clip_policy.to(device)
ref_policy = ref_policy.to(device)


total_samples = len(train_dataloader.dataset)
batch_size = train_dataloader.batch_size
iters_per_epoch = total_samples // batch_size

# define beta moving average
beta_dist = sp.stats.beta(config['beta_rate'], config['beta_rate'])
total_iter = config['num_epochs'] * iters_per_epoch
print(f'Num iters_per_epoch : {iters_per_epoch}') 
weight_func = lambda it: beta_dist.pdf((it + 0.5) / (total_iter + 1))

# optimizer = AdamW(clip_policy.parameters(), lr=learning_rate)
optimizer = torch.optim.Adamax(params=clip_policy.parameters(), lr=config['learning_rate']) 
scheduler = CosineAnnealingLR(optimizer, config['num_epochs'] * iters_per_epoch)
metrics = Metrics()

if method != 'CE':
    policy = GeneralMovingAverage(clip_policy, weight_func)
else :
    policy = clip_policy

if method == 'dpo':
    criterion = DPO_Loss(
            beta=config['beta'],
            label_smoothing=config['label_smoothing'],
            ipo=config['ipo'],
            reference_free=config['reference_free'],
            batch_size=config['batch_size']) 
    
    Model = DPO_Model

elif method == 'kto':
    criterion = KTO_Loss(
            beta=config['beta'],
            desirable_weight=config['desirable_weight'],
            undesirable_weight=config['undesirable_weight'],
            batch_size=config['batch_size'])
    
    Model = DPO_Model

elif method == 'ppo':
    criterion = PPO_Loss(eps_clip=config['eps_clip']) 
    Model = PPO_Model

elif method == 'CE':
    criterion = None
    Model = CE_Model
        
else:
    raise Exception("Please choose a correct method type.")


My_model = Model(policy,
                ref_policy,
                criterion,
                optimizer,
                train_dataloader,
                test_dataloader,
                metrics,
                scheduler,
                path=path,
                device=device,
                prompts=prompts,
                save_model=save_model
                ) 

###############################################  Training   ###############################################
My_model.trainer(config['num_epochs'],My_model.train_dataloader,val_dataloader) 

###############################################  Test   ###############################################
print('='*50)
print('Final inference on Test dataset: \n')
My_model.test(My_model.test_dataloader,prompts) 

print('\nDone!')


sys.stdout.close()
sys.stderr.close()

