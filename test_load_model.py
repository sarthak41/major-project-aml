from train_util import load_model
import torch
import pandas as pd
import json
from data_loading import get_data
from train_util import AddEgoIds, extract_param, add_arange_ids, get_loaders, evaluate_homo, evaluate_hetero
from training import get_model
from torch_geometric.nn import to_hetero, summary

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

args = Arg()

with open('data_config.json', 'r') as config_file:
    data_config = json.load(config_file)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config={
            "epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "model": args.model,
            "data": args.data,
            "num_neighbors": args.num_neighs,
            "lr": extract_param("lr", args),
            "n_hidden": extract_param("n_hidden", args),
            "n_gnn_layers": extract_param("n_gnn_layers", args),
            "loss": "ce",
            "w_ce1": extract_param("w_ce1", args),
            "w_ce2": extract_param("w_ce2", args),
            "dropout": extract_param("dropout", args),
            "final_dropout": extract_param("final_dropout", args),
            "n_heads": extract_param("n_heads", args) if args.model == 'gat' else None
}

if args.ego:
    transform = AddEgoIds()
else:
    transform = None

tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)

add_arange_ids([tr_data, val_data, te_data])
tr_loader, val_loader, te_loader = get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args)

sample_batch = next(iter(tr_loader))
model = get_model(sample_batch, config, args)

model, optimizer, epoch, best_f1 = load_model(model, device, args, config, data_config)
print(epoch, best_f1)