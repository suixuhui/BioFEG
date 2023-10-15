
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from pytorch_transformers.modeling_bert import (
    BertPreTrainedModel,
    BertConfig,
    BertModel,
)

from pytorch_transformers.tokenization_bert import BertTokenizer

from blink.common.ranker_base import BertEncoder, get_model_obj
from blink.common.optimizer import get_bert_optimizer
from torch.autograd import Variable
import torch.autograd as autograd


def load_biencoder(params):
    # Init model
    biencoder = BiEncoderRanker(params)
    return biencoder


class BiEncoderModule(torch.nn.Module):
    def __init__(self, params):
        super(BiEncoderModule, self).__init__()
        ctxt_bert = BertModel.from_pretrained(params["bert_model"])
        cand_bert = BertModel.from_pretrained(params['bert_model'])
        self.context_encoder = BertEncoder(
            ctxt_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.cand_encoder = BertEncoder(
            cand_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.config = ctxt_bert.config

    def forward(
        self,
        token_idx_ctxt,
        segment_idx_ctxt,
        mask_ctxt,
        token_idx_cands,
        segment_idx_cands,
        mask_cands,
    ):
        embedding_ctxt = None
        if token_idx_ctxt is not None:
            embedding_ctxt = self.context_encoder(
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt
            )
        embedding_cands = None
        if token_idx_cands is not None:
            embedding_cands = self.cand_encoder(
                token_idx_cands, segment_idx_cands, mask_cands
            )
        return embedding_ctxt, embedding_cands


class BiEncoderRanker(torch.nn.Module):
    def __init__(self, params, shared=None):
        super(BiEncoderRanker, self).__init__()
        self.params = params
        self.device = 0
        self.n_gpu = torch.cuda.device_count()
        # init tokenizer
        self.NULL_IDX = 0
        self.START_TOKEN = "[CLS]"
        self.END_TOKEN = "[SEP]"
        self.tokenizer = BertTokenizer.from_pretrained(
            params["bert_model"], do_lower_case=params["lowercase"]
        )
        # init model
        self.build_model()
        model_path = params.get("path_to_model", None)
        if model_path is not None:
            self.load_model(model_path)

        self.model = self.model.to(self.device)
        self.data_parallel = params.get("data_parallel")
        if self.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

    def load_model(self, fname, cpu=False):
        if cpu:
            state_dict = torch.load(fname, map_location=lambda storage, location: "cpu")
        else:
            state_dict = torch.load(fname)
        self.model.load_state_dict(state_dict)

    def build_model(self):
        self.model = BiEncoderModule(self.params)

    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = get_model_obj(self.model) 
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

    def get_optimizer(self, optim_states=None, saved_optim_type=None):
        return get_bert_optimizer(
            [self.model],
            self.params["type_optimization"],
            self.params["learning_rate"],
            fp16=self.params.get("fp16"),
        )
 
    def encode_context(self, cands):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        embedding_context, _ = self.model(
            token_idx_cands, segment_idx_cands, mask_cands, None, None, None
        )
        return embedding_context.cpu().detach()

    def encode_candidate(self, cands):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        _, embedding_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )
        return embedding_cands.cpu().detach()
        # TODO: why do we need cpu here?
        # return embedding_cands


    def score_generate_candidate(
        self,
        embedding_ctxt,
        cand_vecs,
        random_negs=True,
        cand_encs=None,  # pre-computed candidate encoding.
    ):
        # Candidate encoding is given, do not need to re-compute
        # Directly return the score of context encoding and candidate encoding
        if cand_encs is not None:
            return embedding_ctxt.mm(cand_encs.t())

        # Train time. We compare with all elements of the batch
        if random_negs:
            token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
                cand_vecs, self.NULL_IDX
            )
            _, embedding_cands = self.model(
                None, None, None, token_idx_cands, segment_idx_cands, mask_cands
            )

            return embedding_ctxt.mm(embedding_cands.t())

        else:
            num_cand = cand_vecs.size(1)
            cand_vecs = cand_vecs.view(-1, cand_vecs.size(-1))
            token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
                cand_vecs, self.NULL_IDX
            )
            _, embedding_cands = self.model(
                None, None, None, token_idx_cands, segment_idx_cands, mask_cands
            )

            embedding_cands = embedding_cands.reshape(-1, num_cand, embedding_ctxt.size(-1))
            embedding_ctxt = embedding_ctxt.unsqueeze(1)
            embedding_cands = embedding_cands.transpose(1, 2)
            scores = torch.bmm(embedding_ctxt, embedding_cands)
            scores = torch.squeeze(scores)
            return scores


    # Score candidates given context input and label input
    # If cand_encs is provided (pre-computed), cand_ves is ignored
    def score_candidate(
        self,
        text_vecs,
        cand_vecs,
        random_negs=True,
        cand_encs=None,  # pre-computed candidate encoding.
    ):
        # Encode contexts first
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
            text_vecs, self.NULL_IDX
        )
        embedding_ctxt, _ = self.model(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt, None, None, None
        )

        # Candidate encoding is given, do not need to re-compute
        # Directly return the score of context encoding and candidate encoding
        if cand_encs is not None:
            return embedding_ctxt.mm(cand_encs.t())

        # Train time. We compare with all elements of the batch
        if random_negs:
            token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
                cand_vecs, self.NULL_IDX
            )
            _, embedding_cands = self.model(
                None, None, None, token_idx_cands, segment_idx_cands, mask_cands
            )

            return embedding_ctxt.mm(embedding_cands.t())

        else:
            num_cand = cand_vecs.size(1)
            cand_vecs = cand_vecs.view(-1, cand_vecs.size(-1))
            token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
                cand_vecs, self.NULL_IDX
            )
            _, embedding_cands = self.model(
                None, None, None, token_idx_cands, segment_idx_cands, mask_cands
            )

            embedding_cands = embedding_cands.reshape(-1, num_cand, embedding_ctxt.size(-1))
            embedding_ctxt = embedding_ctxt.unsqueeze(1)
            embedding_cands = embedding_cands.transpose(1, 2)
            scores = torch.bmm(embedding_ctxt, embedding_cands)
            scores = torch.squeeze(scores)
            return scores

    # label_input -- negatives provided
    # If label_input is None, train on in-batch negatives
    def forward(self, context_input, cand_input, label_input=None, generate=False):
        flag = label_input is None
        if generate:
            scores = self.score_generate_candidate(context_input, cand_input, flag)
        else:
            scores = self.score_candidate(context_input, cand_input, flag)
        bs = scores.size(0)
        if len(scores.shape) == 1:
            scores = torch.unsqueeze(scores, 0)
        if label_input is None:
            target = torch.LongTensor(torch.arange(bs))
            target = target.to(self.device)
            loss = F.cross_entropy(scores, target, reduction="mean")
        else:
            loss = F.cross_entropy(scores, label_input, reduction="mean")
        return loss, scores


def calc_gradient_penalty(discriminator, real_feats, fake_feats, labels, lambda1=10.):
    b = real_feats.size(0)
    alpha = torch.rand(b, 1).to(real_feats.device)
    alpha = alpha.expand(real_feats.size())

    interpolates = alpha * real_feats + ((1 - alpha) * fake_feats)
    interpolates = Variable(interpolates, requires_grad=True)

    if labels.dim() == 1:
        labels = torch.unsqueeze(labels, dim=0)

    disc_interpolates = discriminator(interpolates, labels)

    ones = torch.ones(disc_interpolates.size()).to(real_feats.device)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones, create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda1
    return gradient_penalty


class ConditionalGenerator(nn.Module):
    def __init__(self, noise_size=768, label_size=768, hidden_size=768, output_size=768):
        super(ConditionalGenerator, self).__init__()
        self.fc1 = nn.Linear(noise_size + label_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, noise, label):
        x = torch.cat([noise, label], dim=-1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class ConditionalDiscriminator(nn.Module):
    def __init__(self, feat_size=768, label_size=768, hidden_size=768, output_size=1):
        super(ConditionalDiscriminator, self).__init__()
        self.fc1 = nn.Linear(feat_size + label_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, feat, label):
        x = torch.cat([feat, label], dim=-1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x.squeeze()


class DiscriptionEmbedding(nn.Module):
    def __init__(self, params):
        super(DiscriptionEmbedding, self).__init__()
        cand_bert = BertModel.from_pretrained(params['bert_model'])
        self.cand_encoder = BertEncoder(
            cand_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.NULL_IDX = 0

    def forward(
            self,
            cand_input
    ):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cand_input, self.NULL_IDX
        )
        embedding_cands = None
        if token_idx_cands is not None:
            embedding_cands = self.cand_encoder(
                token_idx_cands, segment_idx_cands, mask_cands
            )
        return embedding_cands


def to_bert_input(token_idx, null_idx):
    """ token_idx is a 2D tensor int.
        return token_idx, segment_idx and mask
    """
    segment_idx = token_idx * 0
    mask = token_idx != null_idx
    # nullify elements in case self.NULL_IDX was not 0
    token_idx = token_idx * mask.long()
    return token_idx, segment_idx, mask
