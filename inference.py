import torch
import pandas as pd
from train_util import extract_param, add_arange_ids, get_loaders, evaluate_model, load_model
from training import get_model
import time
from torch.utils.tensorboard import SummaryWriter

script_start = time.time()

def infer_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config):
    #set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter()

    #define a model config dictionary and wandb logging at the same time
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
    }

    #set the transform if ego ids should be used

    transform = None

    #add the unique ids to later find the seed edges
    add_arange_ids([tr_data, val_data, te_data])

    tr_loader, val_loader, te_loader = get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args)

    #get the model
    sample_batch = next(iter(tr_loader))
    model = get_model(sample_batch, config, args)

    if args.load_ckpt:
        model, optimizer, ckpt_epochs, ckpt_best_f1 = load_model(model, device, args, config, data_config)
    else:
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    te_f1, acc = evaluate_model(te_loader, te_inds, model, te_data, device, args)

    print(f"F1 score: {te_f1:.4f}")
    print(f"Accuracy: {acc:.4f}")