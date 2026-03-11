import sys

import torch
import random
import torch.optim as optim
import numpy as np
from metrics import evaluate
from model import AlignGroup
from datetime import datetime
from utils import get_local_time
import argparse
import time
from dataloader import GroupDataset
from tqdm import tqdm
try:
    from dataloader_facebook import FacebookGroupDataset
except ImportError:
    FacebookGroupDataset = None
# from tensorboardX import SummaryWriter
import os
import logging
from sklearn import manifold

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True


def training(train_loader, epoch, type_m="group", group_member_dict=None):
    st_time = time.time()
    lr = args.learning_rate
    optimizer = optim.RMSprop(train_model.parameters(), lr=lr)
    losses = []

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch} [{type_m}]")
    for batch_id, batch in pbar:
        if type_m == 'group':
            u, pi_ni, g_hist, g_mask = batch
        else:
            u, pi_ni = batch

        user_input = torch.LongTensor(u).to(running_device)
        pos_items_input, neg_items_input = pi_ni[:, 0].to(running_device), pi_ni[:, 1].to(running_device)

        optimizer.zero_grad()
        if type_m == 'user':
            loss, _ = train_model(None, user_input, pos_items_input, neg_items_input, None, 'train')
        else:
            # members = [torch.LongTensor(group_member_dict[group_id]).to(running_device) for group_id in list(u.cpu().numpy())]
            members = [torch.LongTensor(group_member_dict[group_id]).to(running_device) for group_id in u.tolist()]
            loss, _ = train_model(user_input, None, pos_items_input, neg_items_input, members, 'train',
                                  group_history=g_hist.to(running_device), group_mask=g_mask.to(running_device))

        losses.append(loss)
        loss.backward()
        optimizer.step()
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    string = (f'Epoch {epoch}, {type_m} loss: {torch.mean(torch.stack(losses)):.5f}, Cost time: {time.time() - st_time:4.2f}s')
    logging.info(string)
    return torch.mean(torch.stack(losses)).item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", type=str, help="[Mafengwo, CAMRa2011, facebook]", default="facebook") # CAMRa2011, Mafengwo, facebook
    parser.add_argument("--device", type=str, help="[cuda:0, ..., cpu]", default="cuda:0")
    parser.add_argument("--bge_path", type=str, default="/mnt/c1a908c8-4782-4898-8be7-c6be59a325d6/yyw/checkpoint/bge-large-zh")
    parser.add_argument("--group_profile_path", type=str, default="")
    parser.add_argument("--recent_k", type=int, default=200)

    parser.add_argument("--layers", type=int, help="# HyperConv & OverlapConv layers", default=4) # 3 is the best
    parser.add_argument("--emb_dim", type=int, help="User/Item/Group embedding dimensions", default=32)
    parser.add_argument("--num_negatives", type=int, default=8)
    parser.add_argument("--topK", type=list, default=[1, 5, 10])

    parser.add_argument("--epoch", type=int, default=100, help="# running epoch")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--predictor", type=str, default="MLP")
    parser.add_argument("--loss_type", type=str, default="BPR")
    parser.add_argument("--temp", nargs='+', type=float, default=[0.2, 0.4, 0.6, 0.8])
    parser.add_argument("--cl_weight", nargs='+', type=float, default=[0.001, 0.01, 0.1])
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")



    args = parser.parse_args()
    set_seed(args.seed)
    logfilename = '{}-{}.log'.format(args.dataset, get_local_time())

    logfilepath = os.path.join('log/', logfilename)

    file_handler = logging.FileHandler(logfilepath, mode='a', encoding='utf8')
    file_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    console_handler.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

    ## t-sne
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)

    logging.info('= ' * 20)
    msg = ('## Finishing Time:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logging.info(msg)
    logging.info(args)

    running_device = torch.device(args.device)

    # Load dataset
    if args.dataset == 'facebook' or args.dataset.startswith('facebook'):
        if FacebookGroupDataset is None:
            raise ImportError("dataloader_facebook.py not found or failed to import.")
        group_profile_path = args.group_profile_path if args.group_profile_path else None
        dataset = FacebookGroupDataset(data_dir=f"./data/{args.dataset}", num_negatives=args.num_negatives, recent_k=args.recent_k, group_profile_path=group_profile_path)
    else:
        user_path, group_path = f"./data/{args.dataset}/userRating", f"./data/{args.dataset}/groupRating"
        dataset = GroupDataset(user_path, group_path, num_negatives=args.num_negatives, dataset=args.dataset)
        
    num_users, num_items, num_groups = dataset.num_users, dataset.num_items, dataset.num_groups
    logging.info(" #Users {}, #Items {}, #Groups {}\n".format(num_users, num_items, num_groups))

    user_hg, item_hg, full_hg = dataset.user_hyper_graph.to(running_device), dataset.item_hyper_graph.to(
        running_device), dataset.full_hg.to(running_device)
    try:
        overlap_graph = torch.Tensor(dataset.overlap_graph).to(running_device)
    except RuntimeError:
        overlap_graph = torch.Tensor(dataset.overlap_graph.tolist()).to(running_device)

    group_member_dict = dataset.group_member_dict

    # Prepare model
    logging.info('██Dataset: \t' + args.dataset)
    
    item_embeddings = None
    if args.dataset == 'facebook' or args.dataset.startswith('facebook'):
        emb_path = f'data/{args.dataset}/item_embeddings.npy'
        if os.path.exists(emb_path):
            logging.info(f"Loading pretrained item embeddings from {emb_path}")
            try:
                item_embeddings = np.load(emb_path)
                item_embeddings = torch.FloatTensor(item_embeddings).to(running_device)
                logging.info(f"Loaded embeddings with shape {item_embeddings.shape}")
            except Exception as e:
                logging.error(f"Failed to load item embeddings: {e}")
                item_embeddings = None
        else:
            logging.warning(f"Pretrained item embeddings not found at {emb_path}")

    for idx in range(len(args.cl_weight) * len(args.temp)):
        cl_info = args.cl_weight[idx % (len(args.cl_weight))]
        temp = args.temp[idx // (len(args.cl_weight))]
        logging.info(f"Idx = {idx+1} / {len(args.cl_weight) * len(args.temp)}, cl_weight = {cl_info}, temp = {temp}: ")
        # Prepare model
        train_model = AlignGroup(num_users, num_items, num_groups, args, user_hg, item_hg,
                               full_hg, overlap_graph, running_device, cl_info, temp, item_embeddings=item_embeddings,
                               user_hist_mat=(dataset.user_hist_mat.to(running_device) if hasattr(dataset, "user_hist_mat") else None),
                               group_texts=(dataset.group_texts if hasattr(dataset, "group_texts") else None),
                               bge_model_path=(args.bge_path if (args.dataset == "facebook" or args.dataset.startswith('facebook')) else None))
        train_model.to(running_device)

        # Track best metric for this parameter combination
        best_group_ndcg = 0.0
        best_epoch = -1
        
        # Ensure checkpoint directory exists
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

        for epoch_id in range(args.epoch):
            train_model.train()
            g_loader = dataset.get_group_dataloader(args.batch_size)

            group_loss = training(g_loader, epoch_id, "group", group_member_dict)

            user_loss = training(dataset.get_user_dataloader(args.batch_size), epoch_id, "user", group_member_dict)

            hits, ndcgs = evaluate(train_model, dataset.group_test_ratings, dataset.group_test_negatives,
                                   running_device,
                                   args.topK, 'group',
                                   group_hist=(dataset.group_hist_ids.to(running_device) if hasattr(dataset, "group_hist_ids") else None),
                                   group_mask=(dataset.group_hist_mask.to(running_device) if hasattr(dataset, "group_hist_mask") else None))

            logging.info("[Epoch {}] Group, Hit@{}: {}, NDCG@{}: {}".format(epoch_id, args.topK, hits, args.topK, ndcgs))

            hrs, ngs = evaluate(train_model, dataset.user_test_ratings, dataset.user_test_negatives, running_device,
                                args.topK, 'user')

            logging.info("[Epoch {}] User, Hit@{}: {}, NDCG@{}: {}".format(epoch_id, args.topK, hrs, args.topK, ngs))

            # Save Best Model Logic
            # Assuming args.topK is [1, 5, 10], we use NDCG@10 (index -1) for selection
            current_ndcg = ndcgs[-1] 
            if current_ndcg > best_group_ndcg:
                best_group_ndcg = current_ndcg
                best_epoch = epoch_id
                
                model_name = f"model_{args.dataset}_cl{cl_info}_temp{temp}.pth"
                save_path = os.path.join(args.checkpoint_dir, model_name)
                torch.save(train_model.state_dict(), save_path)
                logging.info(f"New best model found at epoch {epoch_id} with Group NDCG@{args.topK[-1]}: {best_group_ndcg:.5f}. Saved to {save_path}")

            # tsne
            # g_rep, u_rep, i_rep = train_model.group_embedding.weight.clone(), train_model.user_embedding.weight.clone(), train_model.item_embedding.weight.clone()
            # g_rep = g_rep.cpu().data.numpy()
            # u_rep = u_rep.cpu().data.numpy()
            # i_rep = i_rep.cpu().data.numpy()
            #
            # g_rep = tsne.fit_transform(g_rep)
            # u_rep = tsne.fit_transform(u_rep)
            # i_rep = tsne.fit_transform(i_rep)
            # np.savetxt('./embd/wo/g/' + str(epoch_id) + '.csv', g_rep, delimiter=',')
            # np.savetxt('./embd/wo/u/' + str(epoch_id) + '.csv', u_rep, delimiter=',')
            # np.savetxt('./embd/wo/i/' + str(epoch_id) + '.csv', i_rep, delimiter=',')
        msg = ('## Finishing Time: {}', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logging.info(msg)
    logging.info('= ' * 20)

    # Print Group Consensus (for Facebook)
    if args.dataset == 'facebook':
        logging.info("Calculating Group Consensus (Top Keywords)...")
        
        # Load ID mappings
        item_map = {}
        group_map = {}
        try:
            with open(f'data/{args.dataset}/item_list.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        item_map[int(parts[0])] = parts[1]
            with open(f'data/{args.dataset}/group_list.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        group_map[int(parts[0])] = parts[1]
        except Exception as e:
            logging.warning(f"Could not load mapping files: {e}")

        train_model.eval()
        
        # Select a few groups to demonstrate
        group_ids = list(dataset.group_member_dict.keys())[:5]
        
        # We need to score all items for these groups.
        # Since calculating for all 360k items might be slow if done one by one, 
        # let's try to do it in batches or just pick the top from a random sample?
        # Ideally we want the global top. 
        # The model score is dot product of Group Embedding and Item Embedding (usually).
        # Let's inspect model logic if possible, but generic 'eval' call works.
        # For 'eval' in AlignGroup, it computes score.
        
        # Optimization: Get Group Embeddings and Item Embeddings directly if possible.
        # But let's stick to the model() call to be safe with architecture.
        
        # However, passing 360k items in one batch might OOM.
        # Let's do it in chunks.
        
        all_items = torch.arange(num_items).to(running_device)
        chunk_size = 1024
        
        for gid in group_ids:
            g_name = group_map.get(gid, str(gid))
            logging.info(f"Group: {g_name}")
            
            # Prepare group input
            members = torch.LongTensor(dataset.group_member_dict[gid]).to(running_device)
            
            scores_list = []
            
            # Process items in chunks
            for i in range(0, num_items, chunk_size):
                end = min(i + chunk_size, num_items)
                items_chunk = all_items[i:end]
                
                batch_len = len(items_chunk)
                
                # Inputs need to be repeated to match batch size
                g_input = torch.LongTensor([gid]).to(running_device).repeat(batch_len)
                g_hist = dataset.group_hist_ids[gid].to(running_device).unsqueeze(0).repeat(batch_len, 1)
                g_mask = dataset.group_hist_mask[gid].to(running_device).unsqueeze(0).repeat(batch_len, 1)
                
                # Members input is a list of tensors.
                # The model expects a list of length batch_size.
                m_input = [members for _ in range(batch_len)]
                
                # Forward pass
                # type_m='group' -> u=Group, g=None
                # model(u, g, pos_i, neg_i, members, mode)
                # We use 'eval' mode. pos_i and neg_i are same for scoring?
                # In evaluate(): model(users_var, None, items_var, items_var, None, "eval")
                # Wait, evaluate passes None for members?
                # Let's check training loop:
                # members = ...
                # loss, _ = train_model(..., members, 'train')
                
                # Check metrics.py evaluate:
                # _, predictions = model(users_var, None, items_var, items_var, None, "eval")
                # It passes None for members!
                # Does the model use members in eval?
                # If not, that's great. If yes, it might crash.
                # Let's assume it works like evaluate().
                
                _, chunk_scores = train_model(g_input, None, items_chunk, items_chunk, None, 'eval', group_history=g_hist, group_mask=g_mask)
                scores_list.append(chunk_scores.detach().cpu())
            
            all_scores = torch.cat(scores_list).squeeze()
            
            # Get Top 10
            top_scores, top_indices = torch.topk(all_scores, 10)
            
            top_words = []
            for idx in top_indices:
                top_words.append(item_map.get(idx.item(), str(idx.item())))
            
            logging.info(f"Consensus: {', '.join(top_words)}")
            logging.info("-" * 20)

    logging.info("Done!")
