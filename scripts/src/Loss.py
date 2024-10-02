import torch
import torch.nn as nn
from torch.nn import functional as F


def DPO_Loss(beta: float, label_smoothing: float = 0.0, ipo: bool = False,
              reference_free: bool = False, batch_size=64, lambda_=1.0):
    def preference_loss(policy_logps_img,
                        policy_logps_txt,
                        ref_logps_img,
                        ref_logps_txt,
                        labels):
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
          policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
          policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
          reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
          reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
          beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
          label_smoothing: conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)
          ipo: If True, use the IPO loss instead of the DPO loss.
          reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

        Returns:
          A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
          The losses tensor contains the DPO loss for each example in the batch.
          The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        b = batch_size
        
        # Create index tensors
        indices_org = torch.arange(b)
        indices_typo = torch.arange(b, b*2)
        
        # Select elements in parallel
        policy_chosen_logps3 = policy_logps_img[indices_typo ,labels[indices_org]]
        policy_rejected_logps3 = policy_logps_img[indices_typo,labels[indices_typo]]

        reference_chosen_logps3 = ref_logps_img[indices_typo ,labels[indices_org]]
        reference_rejected_logps3 = ref_logps_img[indices_typo,labels[indices_typo]]
    
        
        loss3 = DPO(policy_chosen_logps3,
            policy_rejected_logps3,
            reference_chosen_logps3,
            reference_rejected_logps3)

        
        kl_loss = mean_kl_divergence(policy_logps_img[indices_org],ref_logps_img[indices_org])

        total_loss = loss3 + lambda_*kl_loss 

        return total_loss 
    
    def mean_kl_divergence(log_p1, log_p2):
        """
        Calculate the mean KL divergence between log_p1 and log_p2 for a minibatch of samples.
        
        Args:
        - log_p1 (torch.Tensor): Tensor of log probabilities with shape (batch_size, 101)
        - log_p2 (torch.Tensor): Tensor of log probabilities with shape (batch_size, 101)
        
        Returns:
        - mean_kl (torch.Tensor): Mean KL divergence for the minibatch.
        """
        # Ensure the input tensors are of the same shape
        assert log_p1.shape == log_p2.shape, "log_p1 and log_p2 must have the same shape"

        # Calculate the KL divergence for each sample in the batch
        kl_div = F.kl_div(log_p1, log_p2, reduction='batchmean', log_target=True)

        # Return the mean KL divergence
        return kl_div


    def DPO(policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,):
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        # print(pi_logratios.shape, ref_logratios.shape)
        if reference_free:
            ref_logratios = 0

        logits = (pi_logratios - ref_logratios).squeeze(-1)  # also known as h_{\pi_\theta}^{y_w,y_l}
        
        if ipo:
            losses = (logits - 1/(2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        else:
          # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
            losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing

        # chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
        # rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses.mean()
    return preference_loss

def PPO_Loss(eps_clip=0.2, typo_weight=2,device='cuda'):
    def compute_rewards(indices, types):
        """
        Vectorized computation of rewards based on indices and types.

        Args:
            indices (torch.Tensor): Tensor of indices.
            types (torch.Tensor): Tensor of types.
            typo_weight (float): The typo weight to be used in the reward function.

        Returns:
            torch.Tensor: Computed rewards.

        Namely:
            if type_:
                if index == 1:
                    return 1.0
                else:
                    return -1.0
            else:
                if index == 1:
                    return 1.0 * typo_weight
                elif index == 2:
                    return -1.0 * typo_weight
                else:
                    return -1.0
        """
        rewards = torch.zeros_like(indices, dtype=torch.float)

        # Type-specific rewards for y1, y2, y3
        rewards[(indices == 1) & types] = 1.0
        rewards[(indices == 1) & ~types] = 1.5#1.0 * typo_weight
        rewards[(indices == 2) & types] = 0.1
        rewards[(indices == 2) & ~types] = 0.001#-1.0 * typo_weight 
        rewards[(indices == 3)] = 0.1 

        return rewards.to(device)

    def preference_loss(policy_logps, old_log_probs, labels):
        """
        Compute the PPO loss for the given policy and old log probabilities and labels in a parallelized manner.

        Args:
            policy_logps (torch.Tensor): The policy log probabilities tensor of shape (batch_size, 101).
            old_log_probs (torch.Tensor): The old log probabilities tensor of shape (batch_size, 101).
            labels (list of tuples): List of tuples (y1, y2, y3, type_).
            eps_clip (float): The epsilon clipping value for PPO.
            typo_weight (float): The typo weight to be used in the reward function.

        Returns:
            torch.Tensor: The computed PPO loss.
        """
        # Extract y1, y2, y3, and type_ from labels
        y1, y2, y3, types = labels# zip(*labels)

        # Create indices tensors for y1, y2, y3
        batch_indices = torch.arange(policy_logps.size(0))

        # Compute the ratios for y1, y2, y3
        ratios_y1 = (policy_logps[batch_indices, y1] - old_log_probs[batch_indices, y1]).exp()
        ratios_y2 = (policy_logps[batch_indices, y2] - old_log_probs[batch_indices, y2]).exp()
        ratios_y3 = (policy_logps[batch_indices, y3] - old_log_probs[batch_indices, y3]).exp()

        # Compute rewards for y1, y2, y3
        rewards_y1 = compute_rewards(torch.ones_like(batch_indices), types)
        rewards_y2 = compute_rewards(torch.ones_like(batch_indices) * 2, types)
        rewards_y3 = compute_rewards(torch.ones_like(batch_indices) * 3, types)

        # Compute the surrogate and clipped surrogate losses for y1, y2, y3
        surr_loss_y1 = ratios_y1 * rewards_y1
        clipped_surr_loss_y1 = torch.clamp(ratios_y1, 1.0 - eps_clip, 1.0 + eps_clip) * rewards_y1

        surr_loss_y2 = ratios_y2 * rewards_y2
        clipped_surr_loss_y2 = torch.clamp(ratios_y2, 1.0 - eps_clip, 1.0 + eps_clip) * rewards_y2

        surr_loss_y3 = ratios_y3 * rewards_y3
        clipped_surr_loss_y3 = torch.clamp(ratios_y3, 1.0 - eps_clip, 1.0 + eps_clip) * rewards_y3
        
        # Compute the PPO losses for y1, y2, y3 and take their mean
        ppo_loss_y1 = -torch.min(surr_loss_y1, clipped_surr_loss_y1).mean()
        ppo_loss_y2 = -torch.min(surr_loss_y2, clipped_surr_loss_y2).mean()
        ppo_loss_y3 = -torch.min(surr_loss_y3, clipped_surr_loss_y3).mean()

        mean_ppo_loss = (ppo_loss_y1 + ppo_loss_y2 + ppo_loss_y3) / 3.0

        return mean_ppo_loss

    return preference_loss

def KTO_Loss(beta: float, desirable_weight=1.0,
              undesirable_weight=1.0, batch_size=64, lambda_=1.0):
    def preference_loss(policy_logps_img,
                        policy_logps_txt,
                        ref_logps_img,
                        ref_logps_txt,
                        labels):
        
        b = batch_size
        
        # Create index tensors
        indices_org = torch.arange(b)
        indices_typo = torch.arange(b, b*2)
        
        # Select elements in parallel
        policy_chosen_logps3 = policy_logps_img[indices_typo ,labels[indices_org]]
        policy_rejected_logps3 = policy_logps_img[indices_typo,labels[indices_typo]]

        reference_chosen_logps3 = ref_logps_img[indices_typo ,labels[indices_org]]
        reference_rejected_logps3 = ref_logps_img[indices_typo,labels[indices_typo]]

        
        kl_loss = mean_kl_divergence(policy_logps_img[indices_org],ref_logps_img[indices_org])
    
        return kto_loss(
            policy_chosen_logps3,
            policy_rejected_logps3,
            reference_chosen_logps3,
            reference_rejected_logps3,
            kl_loss
        ) + lambda_*kl_loss
    
    
    def mean_kl_divergence(log_p1, log_p2):
        """
        Calculate the mean KL divergence between log_p1 and log_p2 for a minibatch of samples.
        
        Args:
        - log_p1 (torch.Tensor): Tensor of log probabilities with shape (batch_size, 101)
        - log_p2 (torch.Tensor): Tensor of log probabilities with shape (batch_size, 101)
        
        Returns:
        - mean_kl (torch.Tensor): Mean KL divergence for the minibatch.
        """
        # Ensure the input tensors are of the same shape
        assert log_p1.shape == log_p2.shape, "log_p1 and log_p2 must have the same shape"

        # Calculate the KL divergence for each sample in the batch
        kl_div = F.kl_div(log_p1, log_p2, reduction='batchmean', log_target=True)

        # Return the mean KL divergence
        return kl_div
    

    def kto_loss(
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        kl):
        """Compute the KTO loss for a batch of policy and reference model log probabilities."""
        KL = kl.detach() 

        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        chosen_losses = 1 - F.sigmoid(beta * (chosen_logratios - KL))
        chosen_rewards = beta * chosen_logratios.detach()

        rejected_logratios = policy_rejected_logps - reference_rejected_logps
        rejected_losses = 1 - F.sigmoid(beta * (KL - rejected_logratios))
        rejected_rewards = beta * rejected_logratios.detach() 

        losses = torch.cat(
            (desirable_weight * chosen_losses, undesirable_weight * rejected_losses),
            dim=0,
        )

        return losses.mean() # , chosen_rewards, rejected_rewards, KL

    return preference_loss
