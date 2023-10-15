

import argparse
import json
import logging
import os
import torch
from tqdm import tqdm

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from pytorch_transformers.tokenization_bert import BertTokenizer

from blink.biencoder.biencoder import BiEncoderRanker, ConditionalGenerator, ConditionalDiscriminator, DiscriptionEmbedding
import blink.biencoder.data_process as data
import blink.biencoder.nn_prediction as nnquery
import blink.candidate_ranking.utils as utils
from blink.biencoder.zeshel_utils import WORLDS, load_entity_dict_zeshel, Stats
from blink.common.params import BlinkParser
from blink.biencoder.train_gan import generate


def load_generate_entity_dict(logger, params, is_zeshel):
    if is_zeshel:
        return load_entity_dict_zeshel(logger, params)

    path = params.get("entity_dict_path", None)
    assert path is not None, "Error! entity_dict_path is empty."

    train_samples = utils.read_dataset("train", params["data_path"])
    ids = set()
    for sample in train_samples:
        ids.add(int(sample["label_id"]))

    entity_list = []
    label_ids = []
    logger.info("Loading entity description from path: " + path)
    with open(path, 'rt') as f:
        it = 0
        for line in f:
            if it in ids:
                it += 1
                continue
            sample = json.loads(line.rstrip())
            title = sample['title']
            text = sample.get("text", "").strip()
            entity_list.append((title, text))
            label_ids.append(it)
            it += 1
            if params["debug"] and len(entity_list) > 200:
                break

    return entity_list, label_ids


# zeshel version of get candidate_pool_tensor
def get_candidate_pool_tensor_zeshel(
        entity_dict,
        tokenizer,
        max_seq_length,
        logger,
):
    candidate_pool = {}
    for src in range(len(WORLDS)):
        if entity_dict.get(src, None) is None:
            continue
        logger.info("Get candidate desc to id for pool %s" % WORLDS[src])
        candidate_pool[src] = get_candidate_pool_tensor(
            entity_dict[src],
            tokenizer,
            max_seq_length,
            logger,
        )

    return candidate_pool


def get_candidate_pool_tensor_helper(
        entity_desc_list,
        tokenizer,
        max_seq_length,
        logger,
        is_zeshel,
):
    if is_zeshel:
        return get_candidate_pool_tensor_zeshel(
            entity_desc_list,
            tokenizer,
            max_seq_length,
            logger,
        )
    else:
        return get_candidate_pool_tensor(
            entity_desc_list,
            tokenizer,
            max_seq_length,
            logger,
        )


def get_candidate_pool_tensor(
        entity_desc_list,
        tokenizer,
        max_seq_length,
        logger,
):
    # TODO: add multiple thread process
    logger.info("Convert candidate text to id")
    cand_pool = []
    for entity_desc in tqdm(entity_desc_list):
        if type(entity_desc) is tuple:
            title, entity_text = entity_desc
        else:
            title = None
            entity_text = entity_desc

        rep = data.get_candidate_representation(
            entity_text,
            tokenizer,
            max_seq_length,
            title,
        )
        cand_pool.append(rep["ids"])

    cand_pool = torch.LongTensor(cand_pool)
    return cand_pool


def candidate_pool_generation(
        tokenizer,
        params,
        logger,
):
    candidate_pool = None
    label_ids = None
    is_zeshel = params.get("zeshel", None)

    if candidate_pool is None:
        # compute candidate pool from entity list
        entity_desc_list, label_ids = load_generate_entity_dict(logger, params, is_zeshel)
        candidate_pool = get_candidate_pool_tensor_helper(
            entity_desc_list,
            tokenizer,
            params["max_cand_length"],
            logger,
            is_zeshel,
        )
    label_ids = torch.LongTensor(label_ids)

    return candidate_pool, label_ids


def main(params):
    output_path = params["output_path"]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logger = utils.get_logger(params["output_path"])

    # Init model
    reranker = BiEncoderRanker(params)
    tokenizer = reranker.tokenizer
    model = reranker.model
    device = reranker.device

    generator = ConditionalGenerator()
    discriminator = ConditionalDiscriminator()
    entity_embedding = DiscriptionEmbedding(params)
    generator.cuda(device)
    discriminator.cuda(device)
    entity_embedding.cuda(device)

    gan_states = torch.load(params["gan_path"], map_location=lambda storage, loc: storage)
    generator.load_state_dict(gan_states["generator"])
    discriminator.load_state_dict(gan_states["discriminator"])
    entity_embedding.load_state_dict(gan_states["entity_embedding"])

    generate_candidate_pool, label_ids_pool = candidate_pool_generation(
        tokenizer,
        params,
        logger,
    )

    tensor_data = TensorDataset(generate_candidate_pool, label_ids_pool)

    generate_sampler = SequentialSampler(tensor_data)
    generate_dataloader = DataLoader(
        tensor_data,
        sampler=generate_sampler,
        batch_size=params["eval_batch_size"]
    )

    if params["silent"]:
        iter_ = generate_dataloader
    else:
        iter_ = tqdm(generate_dataloader)

    nn_features = []
    nn_candidates = []

    save_results = params.get("save_topk_result")
    for step, batch in enumerate(iter_):
        batch = tuple(t.to(device) for t in batch)
        context_input, label_ids = batch

        generate_feats = generate(generator, entity_embedding, context_input)
        generate_feats = torch.relu(generate_feats)
        for i in range(context_input.size(0)):
            nn_features.append(generate_feats[i].detach().cpu())
            nn_candidates.append(context_input[i].detach().cpu())

    nn_data = {
        'generate_feats': nn_features,
        'candidate_vecs': nn_candidates,
    }


    if save_results:
        save_data_dir = os.path.join(
            params['output_path'],
            "candidates_50",
        )
        if not os.path.exists(save_data_dir):
            os.makedirs(save_data_dir)
        save_data_path = os.path.join(save_data_dir, "generate.t7")
        torch.save(nn_data, save_data_path)


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_eval_args()

    args = parser.parse_args()
    print(args)

    params = args.__dict__

    main(params)
