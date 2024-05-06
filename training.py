import torch
import time
import tqdm
from sklearn.metrics import f1_score, accuracy_score
from train_util import extract_param, add_arange_ids, get_loaders, evaluate_model, save_model, load_model
from models import GIN, PNA
from torch_geometric.nn import summary
from torch_geometric.utils import degree
from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter

import logging

def train_model(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, ckpt_epochs, ckpt_best_f1, loss_fn, args, config, device, val_data, te_data, data_config):
    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter()

    best_val_f1 = ckpt_best_f1
    for epoch in range(config['epochs']):
        print(f"\n\n\nEpoch {ckpt_epochs+epoch+1}\n")
        total_loss = total_examples = 0
        preds = []
        ground_truths = []
        node_embeddings_all = []
        for batch in tqdm.tqdm(tr_loader, disable=not args.tqdm):
            optimizer.zero_grad()
            #select the seed edges from which the batch was created
            inds = tr_inds.detach().cpu()
            batch_edge_inds = inds[batch.input_id.detach().cpu()]
            batch_edge_ids = tr_loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
            mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)

            #remove the unique edge id from the edge features, as it's no longer needed
            batch.edge_attr = batch.edge_attr[:, 1:]

            batch.to(device)
            out, node_embeddings = model(batch.x, batch.edge_index, batch.edge_attr)
            pred = out[mask]
            ground_truth = batch.y[mask]
            preds.append(pred.argmax(dim=-1))
            ground_truths.append(ground_truth)
            # node_embeddings_all.append(node_embeddings)
            loss = loss_fn(pred, ground_truth)

            loss.backward()
            optimizer.step()

            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()

        pred = torch.cat(preds, dim=0).detach().cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).detach().cpu().numpy()
        print(ground_truth.shape, pred.shape)
        f1 = f1_score(ground_truth, pred)
        acc = accuracy_score(ground_truth, pred)
        writer.add_scalar('f1/train', f1, epoch)  # Log training F1 score
        writer.add_scalar('accuracy/train', acc, epoch)
        logging.info(f'Train F1: {f1:.4f}')
        logging.info(f"Train accuracy: {acc:.4f}\n")

        # evaluate
        if (ckpt_epochs+epoch) != 0 and (ckpt_epochs+epoch+1) % 5 == 0:
            val_f1, val_acc = evaluate_model(val_loader, val_inds, model, val_data, device, args)
            te_f1, te_acc = evaluate_model(te_loader, te_inds, model, te_data, device, args)

            writer.add_scalar('f1/validation', val_f1, epoch)  # Log validation F1 score
            writer.add_scalar('f1/test', te_f1, epoch)  # Log test F1 score
            writer.add_scalar("acc/validation", val_acc, epoch)
            writer.add_scalar("acc/test", te_acc, epoch)
            logging.info(f'Validation F1: {val_f1:.4f}, acc: {val_acc:.4f}')
            logging.info(f'Test F1: {te_f1:.4f}, acc: {te_acc:.4f}')

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                writer.add_scalar("best_test_f1", te_f1, epoch)
                if args.save_model:
                    save_model(model, optimizer, ckpt_epochs+epoch, val_f1, args, data_config)
                    print(f"Saved model with f1 score: {val_f1}")

    writer.close()  # Close the SummaryWriter
    return model



def get_model(sample_batch, config, args):
    n_feats = sample_batch.x.shape[1]
    e_dim = (sample_batch.edge_attr.shape[1] - 1)

    if args.model == "gin":
        model = GIN(
                num_features=n_feats, num_gnn_layers=config['n_gnn_layers'], n_classes=2,
                n_hidden=round(config['n_hidden']), residual=False, edge_updates=False, edge_dim=e_dim,
                dropout=config['dropout'], final_dropout=config['final_dropout']
                )
    elif args.model == "pna":
        d = degree(sample_batch.edge_index[1], dtype=torch.long)
        deg = torch.bincount(d, minlength=1)
        model = PNA(
            num_features=n_feats, num_gnn_layers=config['n_gnn_layers'], n_classes=2,
            n_hidden=round(config['n_hidden']), edge_updates=False, edge_dim=e_dim,
            dropout=config['dropout'], deg=deg, final_dropout=config['final_dropout']
            )

    return model

import torch
from torch.utils.tensorboard import SummaryWriter
import tqdm
import logging

def train_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config):
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize TensorBoard writer
    writer = SummaryWriter()

    # Define model configuration
    config = {
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


    # Add unique IDs to find seed edges later
    add_arange_ids([tr_data, val_data, te_data])

    transform = None

    tr_loader, val_loader, te_loader = get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args)

    # Get the model
    sample_batch = next(iter(tr_loader))
    model = get_model(sample_batch, config, args)

    if args.load_ckpt:
        model, optimizer, ckpt_epochs, ckpt_best_f1 = load_model(model, device, args, config, data_config)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    sample_batch.to(device)
    sample_x = sample_batch.x
    sample_edge_index = sample_batch.edge_index

    sample_batch.edge_attr = sample_batch.edge_attr[:, 1:]
    sample_edge_attr = sample_batch.edge_attr
    logging.info(summary(model, sample_x, sample_edge_index, sample_edge_attr))

    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([config['w_ce1'], config['w_ce2']]).to(device))

    model = train_model(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, ckpt_epochs, ckpt_best_f1, loss_fn, args, config, device, val_data, te_data, data_config)

    # Close TensorBoard writer
    writer.close()

    return model
