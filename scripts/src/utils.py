class Metrics:
    def __init__(self):
        self.losses = []
        self.chosen_rewards = []
        self.rejected_rewards = []
        self.reward_dif = []
        self.test_acc = []
        self.val_acc = [] 

    def add(self, loss, chosen_reward=0, rejected_reward=0):
        self.losses.append(loss)
        self.chosen_rewards.append(chosen_reward)
        self.rejected_rewards.append(rejected_reward)
        self.reward_dif.append(chosen_reward - rejected_reward)
        
    def add_test_acc(self,item):
        self.test_acc.append(item)

    def add_val_acc(self,item):
        self.val_acc.append(item) 

    def add_loss(self,item):
        self.losses.append(item) 
