import torch
import tqdm
from typing import Union
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import f1_score, accuracy_score
import json
import os

def extract_param(parameter_name: str, args) -> float:
    """
    Extract the value of the specified parameter for the given model.

    Args:
    - parameter_name (str): Name of the parameter (e.g., "lr").
    - args (argparser): Arguments given to this specific run.

    Returns:
    - float: Value of the specified parameter.
    """
    file_path = './model_settings.json'
    with open(file_path, "r") as file:
        data = json.load(file)

    return data.get(args.model, {}).get("params", {}).get(parameter_name, None)

def add_arange_ids(data_list):
    '''
    Add the index as an id to the edge features to find seed edges in training, validation and testing.

    Args:
    - data_list (str): List of tr_data, val_data and te_data.
    '''
    for data in data_list:
        data.edge_attr = torch.cat([torch.arange(data.edge_attr.shape[0]).view(-1, 1), data.edge_attr], dim=1)

def get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args):

    tr_loader =  LinkNeighborLoader(tr_data, num_neighbors=args.num_neighs, batch_size=args.batch_size, shuffle=True, transform=transform)
    val_loader = LinkNeighborLoader(val_data,num_neighbors=args.num_neighs, edge_label_index=val_data.edge_index[:, val_inds],
                                        edge_label=val_data.y[val_inds], batch_size=args.batch_size, shuffle=False, transform=transform)
    te_loader =  LinkNeighborLoader(te_data,num_neighbors=args.num_neighs, edge_label_index=te_data.edge_index[:, te_inds],
                                edge_label=te_data.y[te_inds], batch_size=args.batch_size, shuffle=False, transform=transform)

    return tr_loader, val_loader, te_loader

@torch.no_grad()
def evaluate_model(loader, inds, model, data, device, args):
    '''Evaluates the model performane for homogenous graph data.'''
    preds = []
    ground_truths = []
    for batch in tqdm.tqdm(loader, disable=not args.tqdm):
        #select the seed edges from which the batch was created
        inds = inds.detach().cpu()
        batch_edge_inds = inds[batch.input_id.detach().cpu()]
        batch_edge_ids = loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
        mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)

        #add the seed edges that have not been sampled to the batch
        missing = ~torch.isin(batch_edge_ids, batch.edge_attr[:, 0].detach().cpu())

        if missing.sum() != 0 and (args.data == 'Small_J' or args.data == 'Small_Q'):
            missing_ids = batch_edge_ids[missing].int()
            n_ids = batch.n_id
            add_edge_index = data.edge_index[:, missing_ids].detach().clone()
            node_mapping = {value.item(): idx for idx, value in enumerate(n_ids)}
            add_edge_index = torch.tensor([[node_mapping[val.item()] for val in row] for row in add_edge_index])
            add_edge_attr = data.edge_attr[missing_ids, :].detach().clone()
            add_y = data.y[missing_ids].detach().clone()

            batch.edge_index = torch.cat((batch.edge_index, add_edge_index), 1)
            batch.edge_attr = torch.cat((batch.edge_attr, add_edge_attr), 0)
            batch.y = torch.cat((batch.y, add_y), 0)

            mask = torch.cat((mask, torch.ones(add_y.shape[0], dtype=torch.bool)))

        #remove the unique edge id from the edge features, as it's no longer needed
        batch.edge_attr = batch.edge_attr[:, 1:]

        with torch.no_grad():
            batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            out = out[mask]
            pred = out.argmax(dim=-1)
            preds.append(pred)
            ground_truths.append(batch.y[mask])
    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    f1 = f1_score(ground_truth, pred)
    acc = accuracy_score(ground_truth, pred)

    return f1, acc

def save_model(model, optimizer, epoch, best_f1, args, data_config):
    # Save the model in a dictionary
    torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1
                }, f'./{data_config["paths"]["model_to_save"]}/checkpoint_{args.unique_name}{"" if not args.finetune else "_finetuned"}.tar')

def load_model(model, device, args, config, data_config):
    checkpoint_path = f'.{data_config["paths"]["model_to_load"]}/checkpoint_{args.unique_name}.tar'

    if not os.path.exists(checkpoint_path):
        # If no checkpoint found, return the original model, optimizer, and epoch
        print(f"No checkpoint found for {args.unique_name}, starting training from 0...\n\n")
        return model, torch.optim.Adam(model.parameters(), lr=config["lr"]), 0, 0

    checkpoint = torch.load(checkpoint_path)
    epoch = checkpoint["epoch"]
    best_f1 = checkpoint["best_f1"] if "best_f1" in checkpoint else 0
    print(f"\n\nCheckpoint found for {args.unique_name} , starting from epoch {epoch+1} (f1: {best_f1:.4f})\n\n")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer, epoch, best_f1