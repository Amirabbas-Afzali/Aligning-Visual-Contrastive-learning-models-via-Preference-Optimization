# config for fine-tuning Clip model on 1 node of 4X A100 40GB

class Train_Config:
    def __init__(self):
        pass

    def get_dpo_config(self):
        return {
            "learning_rate": 2e-5,  
            "batch_size": 512,
            "warmup_ratio": 0.1,
            "num_epochs": 3,
            "beta": 1.0,
            "label_smoothing": 0.0,
            "ipo": False,
            "reference_free": False,
            "data_split_ratio": 0.7,
            "pretokenize": True,
            "beta_rate": 0.7,  # in beta moving average
            "lambda_": 1.0,
        }

    def get_kto_config(self):
        return {
            "learning_rate": 2e-5,  
            "batch_size": 512,
            "warmup_ratio": 0.1,
            "num_epochs": 3,
            "beta": 1.5,
            'desirable_weight': 1.0,
            'undesirable_weight': 1.0,
            "data_split_ratio": 0.7,
            "pretokenize": True,
            "beta_rate": 0.7,  # in beta moving average
            "lambda_": 0.01,
        }
    
    # def get_ppo_config(self):
    #     return {
    #         "learning_rate": 5e-5,  
    #         "batch_size": 128, 
    #         "warmup_ratio": 0.1,
    #         "num_epochs": 3,
    #         "data_split_ratio": 0.7,
    #         "pretokenize": True,
    #         "use_preprocessed_ppo": True,
    #         "beta_rate": 0.7,  # in beta moving average
    #         'eps_clip':1e-4
    #     }

    def get_ce_config(self):
        return {
            "learning_rate": 2e-5, 
            "batch_size": 512,
            "warmup_ratio": 0.1,
            "num_epochs": 3,
            "data_split_ratio": 0.7,
            "pretokenize": True,
            "beta_rate": 0.7,  # in beta moving average 
        } 



