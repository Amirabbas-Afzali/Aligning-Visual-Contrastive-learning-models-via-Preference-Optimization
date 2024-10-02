import torch
import torch.nn as nn
import torch.nn.init as init
import copy
from torch.nn import functional as F
from tqdm import tqdm
import random
import numpy as np
import os 

def set_seed(seed):
    # Set seed for the random module
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # Set seed for PyTorch's CuDNN backend (optional for consistent performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For reproducibility of OS-level operations
    os.environ['PYTHONHASHSEED'] = str(seed)


class ClipPolicy(nn.Module):
    def __init__(self, clip, processor, num_classes,
                 batch_size,device='cuda'):
        """
        Initializes the ClipPolicy module.

        Args:
            model: CLIP model.
            processor: CLIP processor.
            device: Device to run the model on (default: 'cuda').
        """
        super(ClipPolicy, self).__init__()
        self.clip = clip
        self.processor = processor
        self.device = device
        self.num_classes = num_classes
        self.batch_size = batch_size

    def get_logits(self, outputs, option):
        if option == 'per_image':
            return outputs.logits_per_image
        elif option == 'per_text':
            return outputs.logits_per_text
        elif option == 'both':
            return outputs.logits_per_image, outputs.logits_per_text
        else:
            raise ValueError("Invalid option. Choose from 'per_image', 'per_text', or 'both'.")

    def get_model_loss(self):
        """
        Placeholder for computing the model loss.

        Returns:
            0 as placeholder for loss computation.
        """
        return 0
    
    def forward(self, texts, images, option='per_image', tokenized=True, compute_loss=False, return_emb=False):
        if isinstance(images, torch.Tensor):
            images = images.to(self.device)
            
            # assert images.shape[0] == self.batch_size*2, 'Dataparallel has a problem!'
        
        if tokenized:
            # Use tokenized text inputs directly
            images = self.processor(images=images, return_tensors="pt", padding=True)
            inputs = {
                'input_ids': texts['input_ids'],
                'attention_mask': texts['attention_mask'],
                'pixel_values': images['pixel_values']
            }
        else:

            inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
            
        for key, value in inputs.items():
            inputs[key] = value.to(self.device)
        outputs = self.clip(**inputs)
        
        if option == 'both':
            logits_per_image, logits_per_text = self.get_logits(outputs, option)
            # for image
            probs_per_image = logits_per_image.softmax(dim=1)
            logprobs_per_image = logits_per_image.log_softmax(dim=1)

            # for text
            # Create the index tensor for the first part of the pairs
            index1 = torch.arange(self.batch_size).unsqueeze(0).repeat(self.num_classes, 1)

            # Create the index tensor for the second part of the pairs
            index2 = torch.arange(self.batch_size, self.batch_size*2).unsqueeze(0).repeat(self.num_classes, 1)

            # Stack the indices to create pairs, result shape will be [101, 64, 2]
            pairs = torch.stack((index1, index2), dim=2).to(self.device)

            # Gather the logits using the pairs indices
            logits_pairs = torch.stack([logits_per_text.gather(1, pairs[:,:,0]),
                                        logits_per_text.gather(1, pairs[:,:,1])], dim=2)

            # Apply log-softmax to the last dimension of the pairs
            logprobs_per_text = F.log_softmax(logits_pairs, dim=2) # with shape [101, 64, 2]
            
            loss = self.get_model_loss() if compute_loss else None
            out = ((probs_per_image, logprobs_per_image, logits_per_image),
                       (0, logprobs_per_text, logits_per_text),
                       loss) if compute_loss else ((probs_per_image, logprobs_per_image, logits_per_image),
                                                   (0, logprobs_per_text, logits_per_text))
        else:
            logits_per_item = self.get_logits(outputs, option)
            probs = logits_per_item.softmax(dim=1)
            logprobs = logits_per_item.log_softmax(dim=1)
            loss = self.get_model_loss() if compute_loss else None
            out = (probs, logprobs, logits_per_item, loss) if compute_loss else (probs, logprobs, logits_per_item)

        if not return_emb:
            return out  
        else:
            text_embed = outputs.text_embeds.detach()
            image_embed = outputs.image_embeds.detach()
            return text_embed, image_embed
    
    
class ClipPolicyLinearHead(ClipPolicy):
    def __init__(self, model, processor, unitarity_coef, device='cuda'):
        super().__init__(model, processor, device)
        self.linear_head = nn.Linear(model.text_projection.in_features, model.text_projection.out_features // 2, bias=False).to(device)
        init.orthogonal_(self.linear_head.weight)  # Unitary initialization
        self.unitarity_coef = unitarity_coef

    def get_logits(self, outputs, per_image):
        text_embed = self.linear_head(outputs.text_embeds.detach())
        image_embed = self.linear_head(outputs.image_embeds.detach())

        logits_per_text = self.model.logit_scale.exp() * text_embed @ image_embed.t()
        logits_per_image = self.model.logit_scale.exp() * image_embed @ text_embed.t()

        return logits_per_image if per_image else logits_per_text
        
    def get_model_loss(self):
        """
        Computes the model loss.

        Args:
            outputs: CLIP model outputs.

        Returns:
            Frobenius norm of the difference between I - WWt.

        """
        W = self.linear_head.weight
        WWt = torch.mm(W, W.t())
        I = torch.eye(WWt.shape[0], device=WWt.device)
        loss = torch.norm(I - WWt)
        return self.unitarity_coef * loss

class GeneralMovingAverage(object):
    def __init__(self, model, weight_func):
        self.model = model
        self.weight_func = weight_func
        self.iter = 0
        self.weight = weight_func(self.iter)
        self.weight_sum = self.weight
        self.moving_avg = copy.deepcopy(model)

        for param in self.moving_avg.parameters():
            param.requires_grad = False

    def update(self):
        self.iter += 1
        self.weight = self.weight_func(self.iter)  # a_k
        relative_weight = self.weight / self.weight_sum
        for moving_avg_param, param in zip(self.moving_avg.parameters(), self.model.parameters()):
            moving_avg_param.data = (moving_avg_param + relative_weight * param) / (1 + relative_weight)
        self.weight_sum += self.weight

    def __call__(self, x: torch.Tensor):
        return self.moving_avg(x)


    def state_dict(self):
        return self.moving_avg.state_dict()

    def load_state_dict(self, state_dict):
        self.moving_avg.load_state_dict(state_dict)

    @property
    def module(self):
        return self.moving_avg.module

class DPO_Model():
    def __init__(self,clip_policy,
                 ref_policy,
                 criterion,
                 optimizer,
                 train_dataloader,
                 test_dataloader,
                 metrics,
                scheduler,
                path,
                device,
                prompts,
                save_model=True):
        
        self.clip_policy = clip_policy
        self.ref_policy = ref_policy
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.metrics = metrics
        self.scheduler = scheduler
        self.path = path
        self.device = device
        self.prompts = prompts
        self.save_model = save_model 
    
    def load_checkpoint(self,path):
        state_dict = torch.load(path)
        
        self.clip_policy.model.load_state_dict(state_dict['clip_EMA'])
        self.clip_policy.moving_avg.load_state_dict(state_dict['clip_BMA'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.metrics = state_dict['metrics']
        self.scheduler.load_state_dict(state_dict['scheduler'])
        
        print('Loaded!')
            
    def save_checkpoint(self,epoch,path):
        checkpoint = {
            'epoch': epoch + 1,
            'clip_EMA': self.clip_policy.model.state_dict(),
            'clip_BMA': self.clip_policy.moving_avg.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'metrics': self.metrics,
            "scheduler": self.scheduler.state_dict()
            }

        torch.save(checkpoint, f'{path}/CLIP_checkpoint_{epoch+1}.pth')
        print('Saved!')
                    
    def trainer(self,num_epochs, train_dataloader,eval_dataloader):

        texts = self.prompts  # Use the predefined prompts

        total_samples = len(train_dataloader.dataset)
        batch_size = train_dataloader.batch_size
        iters_per_epoch = total_samples // batch_size

        for epoch in range(num_epochs):
            print('-'*50)
            print(f'Start epoch: {epoch+1}')
            print("Learning rate: {:.4e}".format(self.scheduler.get_last_lr()[0]))

            self.clip_policy.model.train()  # Set the model in training model
            total_loss = 0.0
            total_chosen_reward = 0.0
            total_rejected_reward = 0.0

            mini_batch_ctr = 0
            for idx,(true_images, typo_images, true_labels) in enumerate(tqdm(train_dataloader)):
                self.optimizer.zero_grad()  # Zero gradients

                if idx > iters_per_epoch-2:
                    continue

                # if (self.tensorboard) and (idx % 50 == 0):
                    # torch.save(self.clip_policy.model.clip.visual_projection.weight
                    #            , f'/home/ali.rasekh/ambo/final/Tensorboard/tiny/V3/{idx + epoch*iters_per_epoch}.pth')


                # Forward pass for the typographic image
                typo_images = torch.stack([typo_images[0][j] for j in range(len(typo_images[0]))], dim=0)
                typo_labels = torch.stack(typo_labels, dim=1).flatten()
                                
                # concat
                images = torch.cat((true_images, typo_images), dim=0).to(self.device)
                labels = torch.cat((true_labels, typo_labels), dim=0).to(self.device)
                
                (probs_img, logprobs_img, _), (probs_txt, logprobs_txt, _) = self.clip_policy.model(texts, images, option='both')
                (ref_probs_img, ref_logprobs_img, _), (ref_probs_txt, ref_logprobs_txt, _) = self.ref_policy(texts, images, option='both')   
                    
               # Compute DPO loss
                losses = self.criterion(
                    policy_logps_img=logprobs_img, 
                    policy_logps_txt=logprobs_txt,
                    ref_logps_img=ref_logprobs_img, 
                    ref_logps_txt=ref_logprobs_txt,  
                    labels=labels,    
                )
                
                losses.backward()  # Backward pass
                # nn.utils.clip_grad_norm_(self.clip_policy.parameters(), 0.25)
                self.optimizer.step()  # Update model parameters
                self.scheduler.step()  # Update learning rate
                
                self.clip_policy.update()
 
                total_loss += losses.item()
                mini_batch_ctr += 1
#                 total_chosen_reward += chosen_reward.item()
#                 total_rejected_reward += rejected_reward.item() 

            # Print average loss for the epoch
            avg_loss = total_loss / (mini_batch_ctr+1)
#             avg_chosen_reward = total_chosen_reward / len(train_dataloader)
#             avg_rejected_reward = total_rejected_reward / len(train_dataloader)
            print(f"Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}")
            self.metrics.add_loss(avg_loss)
            # print(f'\nValidation results on EMA Clip :')
            # self.validation(eval_dataloader,epoch + 1,self.clip_policy.model)
            print(f'\nValidation results on BMA Clip :')
            self.validation(eval_dataloader,epoch + 1, self.clip_policy.moving_avg)

            if self.save_model:
                self.save_checkpoint(epoch,self.path)
            
        
    def test(self,test_dataloader,prompts):

        self.clip_policy.model.eval()
        self.clip_policy.moving_avg.eval()
        self.ref_policy.eval()

        correct_clip_policy = 0
        correct_clip_policy2 = 0
        total_clip_policy = 0
        
        correct_clip_policy_typo = 0
        correct_clip_policy_typo2 = 0
        total_clip_policy_typo = 0
        
        correct_clip_ref_typo = 0
        total_clip_ref_typo = 0
        
        correct_clip_ref = 0
        total_clip_ref = 0

        texts = prompts  # Use the predefined prompts

        total_samples = len(test_dataloader.dataset)
        batch_size = b = test_dataloader.batch_size
        iters_per_epoch = total_samples // batch_size

        with torch.no_grad():
            for idx,(true_images, typo_images, true_labels) in enumerate(tqdm(test_dataloader)):
                
                if idx > iters_per_epoch-2:
                    continue
                
                # print(len(typo_images),typo_images[0].shape )
                typo_images = torch.stack([typo_images[0][j] for j in range(len(typo_images[0]))], dim=0)
                # typo_labels = torch.stack(typo_labels, dim=1).flatten()
                # concat
                images = torch.cat((true_images, typo_images), dim=0).to(self.device)
                labels = torch.cat((true_labels, true_labels), dim=0).to(self.device)
                
                _, logprobs, _ = self.clip_policy.model(texts, images)
                _, logprobs2, _ = self.clip_policy.moving_avg(texts, images) 
               
                # print(logprobs)
                # assert False, 'ppppppppppp'

                clip_predictions = torch.argmax(logprobs, dim=1)
                clip_predictions2 = torch.argmax(logprobs2, dim=1)
                
                # Update counters for clip_policy_accuracy_typo
                correct_clip_policy_typo += (clip_predictions[b:] == labels[:b]).sum().item()
                correct_clip_policy_typo2 += (clip_predictions2[b:] == labels[:b]).sum().item()
                total_clip_policy_typo += b

                ref_probs, ref_logprobs, ref_logits_per_item = self.ref_policy(texts, images)

                ref_predictions = torch.argmax(ref_logprobs, dim=1)
                

                # Update counters for clip_ref_accuracy_typo
                correct_clip_ref_typo += (ref_predictions[b:] == labels[:b]).sum().item()
                total_clip_ref_typo += b


        # Calculate accuracy
        clip_ref_accuracy_typo = correct_clip_ref_typo / total_clip_ref_typo if total_clip_ref_typo > 0 else 0
        clip_policy_accuracy_typo = correct_clip_policy_typo / total_clip_policy_typo if total_clip_policy_typo > 0 else 0
        clip_policy_accuracy_typo2 = correct_clip_policy_typo2 / total_clip_policy_typo if total_clip_policy_typo > 0 else 0


        print(f"""Accuracy On Typographic Dataset:\n
              EMA Policy : {clip_policy_accuracy_typo:.4f},\n
              BMA Policy : {clip_policy_accuracy_typo2:.4f},\n
              Ref Acc: {clip_ref_accuracy_typo:.4f}""") 
        
        self.metrics.add_test_acc((
            clip_policy_accuracy_typo,
            clip_policy_accuracy_typo2,
            clip_ref_accuracy_typo
        ))
    


    def get_embedding(self,test_dataloader,prompts):

        self.clip_policy.model.eval()
        self.clip_policy.moving_avg.eval()
        self.ref_policy.eval()

        texts = prompts  # Use the predefined prompts

        total_samples = len(test_dataloader.dataset)
        batch_size = b = test_dataloader.batch_size
        iters_per_epoch = total_samples // batch_size

        with torch.no_grad():
            for idx,(true_images, typo_images, true_labels) in enumerate(tqdm(test_dataloader)):
                
                if idx > iters_per_epoch-2:
                    continue

                typo_images = torch.stack([typo_images[0][j] for j in range(len(typo_images[0]))], dim=0)
                typo_labels = torch.stack(typo_labels, dim=1).flatten()
                
                # concat
                images = torch.cat((true_images, typo_images), dim=0).to(self.device)
                labels = torch.cat((true_labels, typo_labels), dim=0).to(self.device)
                
                # text_embed, image_embed = self.clip_policy.model(texts, images, return_emb=True) 
                text_embed, image_embed = self.clip_policy.moving_avg(texts, images , return_emb=True)
                text_embed_ref, image_embed_ref = self.ref_policy(texts, images, return_emb=True)

            return (   # TODO
                text_embed, image_embed,
                text_embed_ref, image_embed_ref
            )

class PPO_Model():
    def __init__(self,clip_policy,
                 ref_policy,
                 criterion,
                 optimizer,
                 train_dataloader,
                 test_dataloader,
                 metrics,
                scheduler,
                path,
                device,
                prompts):
        
        self.clip_policy = clip_policy
        self.ref_policy = ref_policy
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.metrics = metrics
        self.scheduler = scheduler
        self.path = path
        self.device = device
        self.prompts = prompts
    
    def load_checkpoint(self,path):
        state_dict = torch.load(path)
        
        self.clip_policy.model.load_state_dict(state_dict['clip_EMA'])
        self.clip_policy.moving_avg.load_state_dict(state_dict['clip_BMA'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.metrics = state_dict['metrics']
        self.scheduler.load_state_dict(state_dict['scheduler'])
        
        print('Loaded!')
            
    def save_checkpoint(self,epoch,path):
        checkpoint = {
            'epoch': epoch + 1,
            'clip_EMA': self.clip_policy.model.state_dict(),
            'clip_BMA': self.clip_policy.moving_avg.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'metrics': self.metrics,
            "scheduler": self.scheduler.state_dict()
            }

        torch.save(checkpoint, f'{path}/CLIP_checkpoint_{epoch+1}.pth')
        print('Saved!')

                    
    def trainer(self,num_epochs, train_dataloader,eval_dataloader):

        texts = self.prompts

        for epoch in range(num_epochs):
            print('-'*50)
            print(f'Start epoch: {epoch+1}')
            print("Learning rate: {:.4e}".format(self.scheduler.get_last_lr()[0]))

            self.clip_policy.model.train()  # Set the model in training model
            total_loss = 0.0

            for image, y1, y2, y3, type_ in tqdm(train_dataloader):
                self.optimizer.zero_grad()  # Zero gradients
        
                image = image.to(self.device)

                _, logprobs, _ = self.clip_policy.model(texts, image)
                _, ref_logprobs, _ = self.ref_policy(texts, image)
                
               # Compute PPO loss
                losses = self.criterion(
                    policy_logps=logprobs,       # Log probabilities for the current model
                    old_log_probs=ref_logprobs,  # Log probabilities for the reference model 
                    labels=(y1, y2, y3, type_),    
                )
 
                losses.backward()  # Backward pass
                # nn.utils.clip_grad_norm_(self.clip_policy.parameters(), 0.25)
                self.optimizer.step()  # Update model parameters
                self.scheduler.step()  # Update learning rate
                
                self.clip_policy.update()
 
                total_loss += losses.item()


            # Print average loss for the epoch
            avg_loss = total_loss / len(train_dataloader)

            print(f"Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}")
            self.metrics.add_loss(avg_loss)
            print(f'\nValidation results on EMA Clip :')
            self.validation(eval_dataloader,epoch + 1,self.clip_policy.model)
            print(f'\nValidation results on BMA Clip :')
            self.validation(eval_dataloader,epoch + 1, self.clip_policy.moving_avg)

            self.save_checkpoint(epoch,self.path)

            
    def test(self,train_dataloader,prompts):
        
        self.prompts = prompts
        # Measure accuracy of both clip_policy and ref_policy
        (clip_policy_accuracy,
        clip_ref_accuracy,
        clip_policy_accuracy_typo,
        clip_ref_accuracy_typo ) = self.validation(train_dataloader,1,self.clip_policy.model,just_return=True)

        (clip_policy_accuracy2,_,
        clip_policy_accuracy_typo2,_ ) = self.validation(train_dataloader,1,self.clip_policy.moving_avg,just_return=True)
        
        print(f"""Accuracy On Oringinal Dataset:\n
              EMA Policy : {clip_policy_accuracy:.4f},\n
              BMA Policy : {clip_policy_accuracy2:.4f},\n
              Ref Acc: {clip_ref_accuracy:.4f},\n\n
Accuracy On Typographic Dataset:\n
              EMA Policy : {clip_policy_accuracy_typo:.4f},\n
              BMA Policy : {clip_policy_accuracy_typo2:.4f},\n
              Ref Acc: {clip_ref_accuracy_typo:.4f}""") 
        
        self.metrics.add_test_acc((
            clip_policy_accuracy,
            clip_policy_accuracy2,
            clip_ref_accuracy,
            clip_policy_accuracy_typo,
            clip_policy_accuracy_typo2,
            clip_ref_accuracy_typo
        ))
    
    def validation(self,val_dataloader,epoch,model,just_return=False):
        # Measure accuracy of both clip_policy and ref_policy
        model.eval()
        self.ref_policy.eval()

        correct_clip_policy = 0
        total_clip_policy = 0
        
        correct_clip_policy_typo = 0
        total_clip_policy_typo = 0
        
        correct_clip_ref_typo = 0
        total_clip_ref_typo = 0
        
        correct_clip_ref = 0
        total_clip_ref = 0

        texts = self.prompts  # Use the predefined prompts

        with torch.no_grad():
            for image, y1, _, _, type_ in tqdm(val_dataloader):
                
                image = image.to(self.device)
                y1 = y1.to(self.device)
                type_ = type_.to(self.device)

                # Forward pass for the original image
                _, clip_logprobs, _ = model(texts, image)

                # Compute predictions
                clip_predictions = torch.argmax(clip_logprobs, dim=1)

                # Update counters for clip_policy_accuracy
                mask = type_.bool()
                correct_clip_policy += (clip_predictions[mask] == y1[mask]).sum().item()
                total_clip_policy += mask.sum().item()

                # Update counters for clip_policy_accuracy_typo
                mask = ~type_.bool()
                correct_clip_policy_typo += (clip_predictions[mask] == y1[mask]).sum().item()
                total_clip_policy_typo += mask.sum().item()

                if (epoch == 1):
                    _, ref_logprobs, _ = self.ref_policy(texts, image)                    

                    # Compute predictions
                    ref_predictions = torch.argmax(ref_logprobs, dim=1)

                    # Update counters for clip_policy_accuracy
                    mask = type_.bool()
                    correct_clip_ref += (ref_predictions[mask] == y1[mask]).sum().item()
                    total_clip_ref += mask.sum().item()

                    # Update counters for clip_policy_accuracy_typo
                    mask = ~type_.bool()
                    correct_clip_ref_typo += (ref_predictions[mask] == y1[mask]).sum().item()
                    total_clip_ref_typo += mask.sum().item()

        if (epoch == 1) or just_return:
            # Calculate accuracy
            clip_ref_accuracy = correct_clip_ref / total_clip_ref if total_clip_ref > 0 else 0
            clip_ref_accuracy_typo = correct_clip_ref_typo / total_clip_ref_typo if total_clip_ref_typo > 0 else 0

        # Calculate accuracy
        clip_policy_accuracy = correct_clip_policy / total_clip_policy if total_clip_policy > 0 else 0
        clip_policy_accuracy_typo = correct_clip_policy_typo / total_clip_policy_typo if total_clip_policy_typo > 0 else 0

        if not just_return:
            if (epoch == 1):
                print(f"At epoch {epoch}:   Policy Acc: {clip_policy_accuracy:.4f}, Ref Acc: {clip_ref_accuracy:.4f}, Policy typo Acc: {clip_policy_accuracy_typo:.4f}, Ref typo Acc: {clip_ref_accuracy_typo:.4f}")
            else:
                print(f"At epoch {epoch}:   Policy Acc: {clip_policy_accuracy:.4f}, Policy typo Acc: {clip_policy_accuracy_typo:.4f}")


            self.metrics.add_val_acc((
                epoch,
                clip_policy_accuracy,
                clip_policy_accuracy_typo
            ))
        else:
            return (
            clip_policy_accuracy,
            clip_ref_accuracy,
            clip_policy_accuracy_typo,
            clip_ref_accuracy_typo
            )

class CE_Model(): 
    def __init__(self,clip_policy,
                 ref_policy,
                 criterion,
                 optimizer,
                 train_dataloader,
                 test_dataloader,
                 metrics,
                scheduler,
                path,
                device,
                prompts,
                save_model=True):
        
        self.clip_policy = clip_policy
        self.ref_policy = ref_policy
        self.criterion = criterion
        self.path = path
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.metrics = metrics
        self.scheduler = scheduler
        self.device = device
        self.prompts = prompts
        self.num_typographic = 1
        self.save_model = save_model
    
    def load_checkpoint(self,path):
        state_dict = torch.load(path)
        
        self.clip_policy.load_state_dict(state_dict['clip_CE'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.metrics = state_dict['metrics']
        self.scheduler.load_state_dict(state_dict['scheduler'])
        
        print('Loaded!')
            
    def save_checkpoint(self,epoch,path):
        checkpoint = {
            'epoch': epoch + 1,
            'clip_CE': self.clip_policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'metrics': self.metrics,
            "scheduler": self.scheduler.state_dict()
            }

        torch.save(checkpoint, f'{path}/CLIP_checkpoint_{epoch+1}.pth')
        print('Saved!')
                    
    def trainer(self,num_epochs, train_dataloader,eval_dataloader):

        texts = self.prompts  # Use the predefined prompts
        self.num_typographic = 1 # TODO
        
        total_samples = len(train_dataloader.dataset)
        batch_size = train_dataloader.batch_size
        iters_per_epoch = total_samples // batch_size


        for epoch in range(num_epochs):
            print('-'*50)
            print(f'Start epoch: {epoch+1}')

            self.clip_policy.train()  # Set the model in training model
            total_loss = 0.0

            for idx, (image, typographic, label, typolabel) in enumerate(tqdm(train_dataloader)):
                self.optimizer.zero_grad()  # Zero gradients

                # Forward pass for the typographic image
                typo = torch.stack([typographic[i][j] for j in range(len(typographic[0])) for i in range(len(typographic))], dim=0)
                label_extended = torch.repeat_interleave(label, self.num_typographic).to(self.device)

                probs, logprobs, logits_per_item = self.clip_policy(texts, typo)  # image or typo 

                # Compute cross-entropy loss for the original image
                loss = F.cross_entropy(logits_per_item, label_extended.to(self.device))

                loss.backward()  # Backward pass
                self.optimizer.step()  # Update model parameters
                self.scheduler.step()  # Update learning rate
                
                total_loss += loss.item()

            # Print average loss for the epoch
            avg_loss = total_loss / len(train_dataloader)

            print(f"Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}")
            self.metrics.add_loss(avg_loss)
            print(f'\nValidation results on CE Clip :')
            self.validation(eval_dataloader,epoch + 1)

            if self.save_model:
                self.save_checkpoint(epoch,self.path)
            

            
    def test(self,train_dataloader,prompts):
        # Measure accuracy of both clip_policy and ref_policy
        self.clip_policy.eval()
        self.ref_policy.eval()

        total_clip_correct = 0
        total_ref_correct = 0
        total_clip_correct_typo = 0
        total_ref_correct_typo = 0
        total_samples = 0

        with torch.no_grad():
            for image, typographic, label, typolabel in train_dataloader:
                texts = prompts  # Use the predefined prompts
                typo = torch.stack([typographic[i][j] for j in range(len(typographic[0])) for i in range(len(typographic))], dim=0)
                label_extended = torch.repeat_interleave(label, self.num_typographic)

                # Forward pass for the original image
                _, clip_logprobs, _ = self.clip_policy(texts, image)
                _, ref_logprobs, _ = self.ref_policy(texts, image)
                _, clip_logprobs_typo, _ = self.clip_policy(texts, typo)
                _, ref_logprobs_typo, _ = self.ref_policy(texts, typo)

                # Compute predictions
                clip_predictions = torch.argmax(clip_logprobs, dim=1)
                ref_predictions = torch.argmax(ref_logprobs, dim=1)
                clip_predictions_typo = torch.argmax(clip_logprobs_typo, dim=1)
                ref_predictions_typo = torch.argmax(ref_logprobs_typo, dim=1)

                # Count correct predictions for each model
                total_clip_correct += (clip_predictions.cpu() == label).sum().item()
                total_ref_correct += (ref_predictions.cpu() == label).sum().item()
                total_clip_correct_typo += (clip_predictions_typo.cpu() == label_extended).sum().item()
                total_ref_correct_typo += (ref_predictions_typo.cpu() == label_extended).sum().item()
                total_samples += label.size(0)

        # Compute accuracy for each model
        clip_policy_accuracy = total_clip_correct / total_samples
        clip_ref_accuracy = total_ref_correct / total_samples
        clip_policy_accuracy_typo = total_clip_correct_typo / total_samples / self.num_typographic 
        clip_ref_accuracy_typo = total_ref_correct_typo / total_samples / self.num_typographic


        print(f"""Accuracy On Oringinal Dataset:\n
              CE Policy : {clip_policy_accuracy:.4f},\n
              Ref Acc: {clip_ref_accuracy:.4f},\n\n
Accuracy On Typographic Dataset:\n
              CE Policy : {clip_policy_accuracy_typo:.4f},\n
              Ref Acc: {clip_ref_accuracy_typo:.4f}""")  
        
        self.metrics.add_test_acc((
            clip_policy_accuracy,
            clip_ref_accuracy,
            clip_policy_accuracy_typo,
            clip_ref_accuracy_typo
        ))
    
    def validation(self,train_dataloader,epoch):
        # Measure accuracy of both clip_policy and ref_policy
        self.clip_policy.eval()

        total_clip_correct = 0
        total_clip_correct_typo = 0
        total_samples = 0

        with torch.no_grad():
            for image, typographic, label, typolabel in tqdm(train_dataloader):
                texts = self.prompts  # Use the predefined prompts
                typo = torch.stack([typographic[i][j] for j in range(len(typographic[0])) for i in range(len(typographic))], dim=0)
                label_extended = torch.repeat_interleave(label, self.num_typographic)

                # Forward pass for the original image
                _, clip_logprobs, _ = self.clip_policy(texts, image)
                _, clip_logprobs_typo, _ = self.clip_policy(texts, typo)

                # Compute predictions
                clip_predictions = torch.argmax(clip_logprobs, dim=1)
                clip_predictions_typo = torch.argmax(clip_logprobs_typo, dim=1)

                # Count correct predictions for each model
                total_clip_correct += (clip_predictions.cpu() == label).sum().item()
                total_clip_correct_typo += (clip_predictions_typo.cpu() == label_extended).sum().item()
                total_samples += label.size(0)

        # Compute accuracy for each model
        clip_policy_accuracy = total_clip_correct / total_samples
        clip_policy_accuracy_typo = total_clip_correct_typo / total_samples / self.num_typographic 

        print(f"At epoch {epoch}:   Policy Acc: {clip_policy_accuracy:.4f}, Policy typo Acc: {clip_policy_accuracy_typo:.4f}")

        self.metrics.add_val_acc((
            epoch,
            clip_policy_accuracy,
            clip_policy_accuracy_typo
        ))

