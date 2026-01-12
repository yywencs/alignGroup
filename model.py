import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

_GROUP_TEXT_BGE_CACHE = {}


class PredictLayer(nn.Module):
    def __init__(self, emb_dim, drop_ratio=0.):
        super(PredictLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(emb_dim, 8),
            nn.LeakyReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.linear(x)


class HyperGraphBasicConvolution(nn.Module):
    def __init__(self, input_dim):
        super(HyperGraphBasicConvolution, self).__init__()
        self.aggregation = nn.Linear(2 * input_dim, input_dim)

    def forward(self, user_emb, item_emb, group_emb, user_hyper_graph, item_hyper_graph, full_hyper):
        user_msg = torch.sparse.mm(user_hyper_graph, user_emb)
        item_msg = torch.sparse.mm(item_hyper_graph, item_emb)
        msg = self.aggregation(torch.cat([user_msg, item_msg], dim=1))
        norm_emb = torch.mm(full_hyper, msg)

        return norm_emb, msg


class HyperGraphConvolution(nn.Module):
    """Hyper-graph Convolution for Member-level hyper-graph"""

    def __init__(self, user_hyper_graph, item_hyper_graph, full_hyper, layers,
                 input_dim, device):
        super(HyperGraphConvolution, self).__init__()
        self.layers = layers
        self.user_hyper, self.item_hyper, self.full_hyper_graph = user_hyper_graph, item_hyper_graph, full_hyper
        self.hgnns = [HyperGraphBasicConvolution(input_dim).to(device) for _ in range(layers)]

    def forward(self, user_emb, item_emb, group_emb, num_users, num_items):
        final_ui = [torch.cat([user_emb, item_emb], dim=0)]
        final_g = [group_emb]
        for i in range(len(self.hgnns)):
            hgnn = self.hgnns[i]
            emb, he_msg = hgnn(user_emb, item_emb, group_emb, self.user_hyper, self.item_hyper, self.full_hyper_graph)
            user_emb, item_emb = torch.split(emb, [num_users, num_items])
            final_ui.append(emb)
            final_g.append(he_msg)

        final_ui = torch.sum(torch.stack(final_ui), dim=0)
        final_g = torch.sum(torch.stack(final_g), dim=0)
        return final_ui, final_g


class DynamicGroupEncoder(nn.Module):
    def __init__(self, emb_dim):
        super(DynamicGroupEncoder, self).__init__()
        self.attn = nn.Linear(emb_dim, 1)

    def forward(self, item_emb, history_ids, history_mask):
        hist_emb = item_emb[history_ids]
        attn_logits = self.attn(hist_emb).squeeze(-1)
        attn_logits = attn_logits.masked_fill(history_mask <= 0, -1e9)
        attn = torch.softmax(attn_logits, dim=-1)
        out = torch.sum(hist_emb * attn.unsqueeze(-1), dim=1)
        return out


class AlignGroup(nn.Module):
    """AlignGroup"""
    def __init__(self, num_users, num_items, num_groups, args, user_hyper_graph, item_hyper_graph,
                 full_hyper, overlap_graph, device, cl_info, temp, item_embeddings=None,
                 user_hist_mat=None, group_texts=None, bge_model_path=None):
        super(AlignGroup, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.num_groups = num_groups

        # Hyper-parameters
        self.emb_dim = args.emb_dim
        self.layers = args.layers
        self.device = args.device
        self.predictor_type = args.predictor

        self.temp = temp
        self.cl_weight = cl_info

        self.overlap_graph = overlap_graph
        self.user_hist_mat = user_hist_mat
        
        # Item Embedding setup
        self.pretrained_item_emb = None
        if item_embeddings is not None:
            self.pretrained_item_emb = item_embeddings # Expected tensor on device
            self.item_projection = nn.Linear(item_embeddings.shape[1], self.emb_dim)
            self.item_embedding = None 
        else:
            self.item_embedding = nn.Embedding(num_items, self.emb_dim)
            nn.init.xavier_uniform_(self.item_embedding.weight)

        self.user_embedding = None
        if self.user_hist_mat is None:
            self.user_embedding = nn.Embedding(num_users, self.emb_dim)
            nn.init.xavier_uniform_(self.user_embedding.weight)

        self.group_embedding = None
        if group_texts is None:
            self.group_embedding = nn.Embedding(num_groups, self.emb_dim)
            nn.init.xavier_uniform_(self.group_embedding.weight)

        self.group_static_bge = None
        self.group_text_projection = None
        self.group_static_weight = None
        if group_texts is not None and bge_model_path is not None:
            group_static_bge = self._encode_texts_bge(group_texts, bge_model_path, device)
            self.group_static_bge = group_static_bge
            self.group_text_projection = nn.Linear(group_static_bge.shape[1], self.emb_dim)
            nn.init.xavier_uniform_(self.group_text_projection.weight)
            self.group_static_weight = nn.Parameter(torch.tensor(0.0))

        if self.item_embedding is not None:
            nn.init.xavier_uniform_(self.item_embedding.weight)

        # Hyper-graph Convolution
        self.hyper_graph_conv = HyperGraphConvolution(user_hyper_graph, item_hyper_graph, full_hyper, self.layers,
                                                      self.emb_dim, device)

        # Prediction Layer
        self.predict = PredictLayer(self.emb_dim)
        self.dynamic_group_encoder = DynamicGroupEncoder(self.emb_dim)

    def forward(self, group_inputs, user_inputs, pos_item_inputs, neg_item_inputs, members, mode, group_history=None, group_mask=None):
        if (group_inputs is not None) and (user_inputs is None):
            return self.group_forward(group_inputs, pos_item_inputs, neg_item_inputs, members, mode, group_history, group_mask)
        else:
            return self.user_forward(user_inputs, pos_item_inputs, neg_item_inputs, members, mode)

    def get_item_embedding(self):
        if self.pretrained_item_emb is not None:
            return self.item_projection(self.pretrained_item_emb)
        else:
            return self.item_embedding.weight

    def _encode_texts_bge(self, texts, model_path, device):
        cache_key = (model_path, tuple(texts))
        cached = _GROUP_TEXT_BGE_CACHE.get(cache_key)
        if cached is not None:
            return cached.to(device)

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        model.eval()
        model = model.to(device)
        batch_size = 64
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                encoded = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
                encoded = {k: v.to(device) for k, v in encoded.items()}
                out = model(**encoded)
                emb = out.last_hidden_state[:, 0]
                emb = F.normalize(emb, p=2, dim=1)
                all_embeddings.append(emb)
        embeddings = torch.cat(all_embeddings, dim=0).detach().cpu()
        _GROUP_TEXT_BGE_CACHE[cache_key] = embeddings
        return embeddings.to(device)

    def _get_user_base_embedding(self, current_item_emb):
        if self.user_hist_mat is not None:
            return torch.sparse.mm(self.user_hist_mat, current_item_emb)
        return self.user_embedding.weight

    def _get_group_base_embedding(self, group_inputs):
        static_emb = None
        if self.group_static_bge is not None and self.group_text_projection is not None:
            static_emb = self.group_text_projection(self.group_static_bge)
            return static_emb[group_inputs]

        if self.group_embedding is not None:
            return self.group_embedding(group_inputs)

        return torch.zeros((len(group_inputs), self.emb_dim), device=group_inputs.device)

    def group_forward(self, group_inputs, pos_item_inputs, neg_item_inputs, members, mode, group_history, group_mask):
        current_item_emb = self.get_item_embedding()
        user_base = self._get_user_base_embedding(current_item_emb)
        group_emb_init = torch.zeros((self.num_groups, self.emb_dim), device=current_item_emb.device)
        
        # Member-level graph computation
        ui_emb, g_emb = self.hyper_graph_conv(user_base, current_item_emb,
                                               group_emb_init, self.num_users, self.num_items)
        u_emb, i_emb = torch.split(ui_emb, [self.num_users, self.num_items])

        i_emb_pos = i_emb[pos_item_inputs]
        i_emb_neg = i_emb[neg_item_inputs]
        dynamic_g = self.dynamic_group_encoder(i_emb, group_history, group_mask)
        static_g = self._get_group_base_embedding(group_inputs)
        if self.group_static_weight is not None:
            w = torch.sigmoid(self.group_static_weight)
            g_use = w * static_g + (1.0 - w) * dynamic_g
        else:
            g_use = dynamic_g
        if mode == 'eval':
            loss, pos_prediction = self.BPR_loss(g_use, i_emb_pos, i_emb_neg)
            return loss, pos_prediction
        else:
            centers = self.get_centers(members, u_emb)
            # cl
            cl_loss = self.InfoNCE(centers, g_use, self.temp)
            bpr_loss, pos_prediction = self.BPR_loss(g_use, i_emb_pos, i_emb_neg)
            loss = bpr_loss + cl_loss * self.cl_weight
            return loss, pos_prediction

    def get_centers(self, members, u_emb):
        centers_list = []
        for member in members:
            embedding_member = torch.index_select(u_emb, 0, member)
            center = self.geometric_group(embedding_member)
            centers_list.append(center)
        return torch.stack(centers_list)

    def geometric_group(self, embedding_member):
        """Geometric bounding and projection for group representation"""
        u_max = torch.max(embedding_member, dim=0).values
        u_min = torch.min(embedding_member, dim=0).values
        center = (u_max + u_min) / 2
        return center

    def BPR_loss(self, g_emb, i_emb_pos, i_emb_neg):
        # For CAMRa2011, we use DOT mode to avoid the dead ReLU
        if self.predictor_type == "MLP":
            pos_prediction = torch.sigmoid(self.predict(g_emb * i_emb_pos))
            neg_prediction = torch.sigmoid(self.predict(g_emb * i_emb_neg))
        else:
            pos_prediction = torch.sum(g_emb * i_emb_pos, dim=-1)
            neg_prediction = torch.sum(g_emb * i_emb_neg, dim=-1)
        loss = torch.mean(torch.nn.functional.softplus(neg_prediction - pos_prediction))
        return loss, pos_prediction

    def InfoNCE(self, view1, view2, temp):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temp)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temp).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def user_forward(self, user_inputs, pos_item_inputs, neg_item_inputs, members, mode):
        current_item_emb = self.get_item_embedding()
        user_base = self._get_user_base_embedding(current_item_emb)
        u_emb = user_base[user_inputs]
        i_emb_pos = current_item_emb[pos_item_inputs]
        i_emb_neg = current_item_emb[neg_item_inputs]
        if self.predictor_type == "MLP":
            pos_prediction = torch.sigmoid(self.predict(u_emb * i_emb_pos))
            neg_prediction = torch.sigmoid(self.predict(u_emb * i_emb_neg))
        else:
            pos_prediction = torch.sum(u_emb * i_emb_pos, dim=-1)
            neg_prediction = torch.sum(u_emb * i_emb_neg, dim=-1)
        loss = torch.mean(torch.nn.functional.softplus(neg_prediction - pos_prediction))
        return loss, pos_prediction
