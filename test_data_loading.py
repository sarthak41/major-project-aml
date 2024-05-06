import json
from data_loading import get_data
from util import logger_setup

with open('data_config.json', 'r') as config_file:
    data_config = json.load(config_file)

class Arg:
  def __init__(self):
    self.seed = 1
    self.n_epochs = 100
    self.batch_size = 8192
    self.num_neighs = [100, 100]
    self.tqdm = True
    self.data = "Small_HI"
    self.model = "pna"
    self.testing = False
    self.save_model = True
    self.unique_name = 'pna1'
    self.finetune = False
    self.inference = False
    self.ports = False
    self.tds = False
    self.ego = False
    self.reverse_mp = False
    self.emlps = False

logger_setup()

args = Arg()
tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)