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
import copy
from accelerate import Accelerator
import scipy as sp
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.Dataset import DatasetHandler
from src.Loss import DPO_Loss, PPO_Loss, KTO_Loss
from src.utils import Metrics 
from src.Models_zh_syn import *
from config.config import Train_Config
import gc 

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
manualSeed = 0
set_seed(manualSeed)
print("Random Seed: ", manualSeed)

save_model = True

method = 'kto'  # options: 'dpo', 'kto', 'CE' 
Test_Dataset = 'sun'     

YOUR_PATH = 'your_path'
###############################################  Dataset   ###############################################
size = 168
preprocessor = Compose([
    Resize(size=size, interpolation=Image.BICUBIC),
    CenterCrop(size=(size, size)),
    PILToTensor(),
])

# Define the object
num_typographic = 1
obj = type('obj', (object,), {'dataset': f'{Test_Dataset}', 'num_typographic': num_typographic})

# Initialize the DatasetHandler
handler = DatasetHandler(obj, preprocessor, data_split_ratio=0.01, batch_size=512, pretokenize=True)

# # Create and save the dataset
# handler.create_data(f"{YOUR_PATH}/datasets/typographic_image_classes_{Test_Dataset}.p")

# Load typographic image classes
handler.load_typographic_image_classes(f"{YOUR_PATH}/datasets/typographic_image_classes_{Test_Dataset}.p")


handler.load_dataset() 
prompts = handler.prepare_prompts() 
handler.split_data_for_zeroshut_datasets()


if method == 'ppo':
    handler.ppo_process(dataset_name=Test_Dataset)

# Get DataLoaders
train_dataloader = None #handler.get_train_dataloader()
test_dataloader = handler.get_test_dataloader()

# Print to verify the setup
print(f"Number of classes: {handler.NUM_CLASSES}")
print(f"Test dataset size: {len(handler.test_subset)}")

###############################################  Define Model   ###############################################
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



def redirect_output(path, log_name, err_name):
    sys.stdout = open(f'{path}/{log_name}', 'w')
    sys.stderr = open(f'{path}/{err_name}', 'w')

def restore_output(original_stdout, original_stderr):
    sys.stdout.close()  # Close current stdout
    sys.stderr.close()  # Close current stderr
    sys.stdout = original_stdout
    sys.stderr = original_stderr

def main(param):
    # Save the original stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    path = f'{YOUR_PATH}/models/beta_search/KTO/param_{param}'
    
    if not os.path.exists(path):
        os.makedirs(path)
        assert False, 'path Error!'
    else:
        print(f"Directory '{path}' already exists.")

    # Redirect output to the log files
    redirect_output(path, 'zeroshut_logs.out', 'zeroshut_errors.err') 

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

    
    iters_per_epoch = 10  # NOT important

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
    else:
        policy = clip_policy

    if method == 'dpo':
        criterion = DPO_Loss(
                beta=config['beta'],
                label_smoothing=config['label_smoothing'],
                ipo=config['ipo'],
                reference_free=config['reference_free'],
                batch_size=config['batch_size'],
                lambda_=config['lambda_'])  
        
        Model = DPO_Model

    elif method == 'kto':
        criterion = KTO_Loss(
                beta=config['beta'],
                desirable_weight=config['desirable_weight'],
                undesirable_weight=config['undesirable_weight'],
                batch_size=config['batch_size'],
                lambda_=config['lambda_']) 
        
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


    cp_path = f'{path}/CLIP_checkpoint_3.pth'
    My_model.load_checkpoint(cp_path)

    ###############################################  Test   ###############################################
    print('='*50)
    print(f'Final inference {method} version on {Test_Dataset} dataset: \n') 
    My_model.test(My_model.test_dataloader, prompts) 

    print('\nDone!')

    # Clean up models after each fold (optional, depending on memory management)
    del My_model, clip_model, clip_processor, clip_policy, ref_policy , optimizer
    torch.cuda.empty_cache()
    gc.collect()

    # Restore the original stdout and stderr
    restore_output(original_stdout, original_stderr)

    # Optionally, redirect to new logs (if needed)
    new_log_path = f'{path}/new_logs'
    if not os.path.exists(new_log_path):
        os.makedirs(new_log_path)
    
    # Redirect to new logs
    redirect_output(new_log_path, 'zeroshut_logs.out', 'zeroshut_errors.err')

    # Final restoration
    restore_output(original_stdout, original_stderr)

if __name__ == "__main__":
    param_list = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5]
    for param in param_list: 
        main(param)

    print('End.')