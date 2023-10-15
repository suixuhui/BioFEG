

import json
import logging
import torch
from tqdm import tqdm

import blink.candidate_ranking.utils as utils
from blink.biencoder.zeshel_utils import WORLDS, Stats
from blink.biencoder.train_gan import generate


def get_topk_predictions(
    reranker,
    train_dataloader,
    candidate_pool,
    cand_encode_list,
    silent,
    logger,
    top_k=10,
    is_zeshel=False,
    save_predictions=False,
):
    reranker.model.eval()
    device = reranker.device
    logger.info("Getting top %d predictions." % top_k)
    if silent:
        iter_ = train_dataloader
    else:
        iter_ = tqdm(train_dataloader)

    nn_context = []
    nn_candidates = []
    nn_labels = []
    nn_worlds = []
    nn_label_ids = []
    nn_mentions_first = []
    nn_mentions_last = []
    stats = {}

    if is_zeshel:
        world_size = len(WORLDS)
    else:
        # only one domain
        world_size = 1
        candidate_pool = [candidate_pool]
        cand_encode_list = [cand_encode_list]

    logger.info("World size : %d" % world_size)

    for i in range(world_size):
        stats[i] = Stats(top_k)
    
    oid = 0
    src = 0
    for step, batch in enumerate(iter_):
        batch = tuple(t.to(device) for t in batch)
        context_input, _, label_ids = batch
        scores = reranker.score_candidate(
            context_input,
            None, 
            cand_encs=cand_encode_list[src].to(device)
        )
        values, indicies = scores.topk(top_k)
        old_src = src
        for i in range(context_input.size(0)):
            oid += 1
            inds = indicies[i]

            pointer = -1
            for j in range(top_k):
                if inds[j].item() == label_ids[i].item():
                    pointer = j
                    break
            stats[src].add(pointer)


            if not save_predictions:
                continue

            if pointer == -1:
                continue
            cur_candidates = candidate_pool[src][inds]
            nn_context.append(context_input[i].cpu().tolist())
            nn_candidates.append(cur_candidates.cpu().tolist())
            nn_labels.append(pointer)
            nn_worlds.append(src)

            # while training biencoder_hard
            # cur_candidates = candidate_pool[src][inds].cpu().tolist()
            # if pointer == -1:
            #     cur_candidates.insert(0, candidate_pool[src][label_ids[i].item()].cpu().tolist())
            #     cur_candidates = cur_candidates[:-1]
            #     pointer = 0
            # nn_context.append(context_input[i].cpu().tolist())
            # nn_candidates.append(cur_candidates)
            # nn_labels.append(pointer)
            # nn_worlds.append(src)

    res = Stats(top_k)
    for src in range(world_size):
        if stats[src].cnt == 0:
            continue
        if is_zeshel:
            logger.info("In world " + WORLDS[src])
        output = stats[src].output()
        logger.info(output)
        res.extend(stats[src])

    logger.info(res.output())

    nn_context = torch.LongTensor(nn_context)
    nn_candidates = torch.LongTensor(nn_candidates)
    nn_labels = torch.LongTensor(nn_labels)
    nn_data = {
        'context_vecs': nn_context,
        'candidate_vecs': nn_candidates,
        'labels': nn_labels,
    }

    if is_zeshel:
        nn_data["worlds"] = torch.LongTensor(nn_worlds)
    
    return nn_data


def get_generate_topk_predictions(
        reranker,
        generator,
        discriminator,
        entity_embedding,
        train_dataloader,
        candidate_pool,
        cand_encode_list,
        silent,
        logger,
        top_k=10,
        is_zeshel=False,
        save_predictions=False,
):
    reranker.model.eval()
    generator.eval()
    discriminator.eval()
    entity_embedding.eval()
    device = reranker.device
    logger.info("Getting top %d predictions." % top_k)
    if silent:
        iter_ = train_dataloader
    else:
        iter_ = tqdm(train_dataloader)

    nn_context = []
    nn_candidates = []
    nn_labels = []
    nn_worlds = []
    stats = {}

    if is_zeshel:
        world_size = len(WORLDS)
    else:
        # only one domain
        world_size = 1
        candidate_pool = [candidate_pool]
        cand_encode_list = [cand_encode_list]

    logger.info("World size : %d" % world_size)

    for i in range(world_size):
        stats[i] = Stats(top_k)

    oid = 0
    src = 0
    for step, batch in enumerate(iter_):
        batch = tuple(t.to(device) for t in batch)
        context_input, label_ids = batch

        generate_feats = generate(generator, entity_embedding, context_input)
        generate_feats = torch.relu(generate_feats)

        scores = reranker.score_generate_candidate(
            generate_feats,
            None,
            cand_encs=cand_encode_list[src].to(device)
        )
        values, indicies = scores.topk(top_k)
        for i in range(context_input.size(0)):
            oid += 1
            inds = indicies[i]

            pointer = -1
            for j in range(top_k):
                if inds[j].item() == label_ids[i].item():
                    pointer = j
                    break
            stats[src].add(pointer)

            if not save_predictions:
                continue

            cur_candidates = candidate_pool[src][inds].cpu().tolist()
            if pointer == -1:
                cur_candidates.insert(0, candidate_pool[src][label_ids[i].item()].cpu().tolist())
                cur_candidates = cur_candidates[:-1]
                pointer = 0
            nn_context.append(generate_feats[i].cpu().tolist())
            nn_candidates.append(cur_candidates)
            nn_labels.append(pointer)
            nn_worlds.append(src)

    res = Stats(top_k)
    for src in range(world_size):
        if stats[src].cnt == 0:
            continue
        if is_zeshel:
            logger.info("In world " + WORLDS[src])
        output = stats[src].output()
        logger.info(output)
        res.extend(stats[src])

    logger.info(res.output())

    nn_context = torch.tensor(nn_context)
    nn_candidates = torch.LongTensor(nn_candidates)
    nn_labels = torch.LongTensor(nn_labels)
    nn_data = {
        'context_vecs': nn_context,
        'candidate_vecs': nn_candidates,
        'labels': nn_labels,
    }

    if is_zeshel:
        nn_data["worlds"] = torch.LongTensor(nn_worlds)

    return nn_data