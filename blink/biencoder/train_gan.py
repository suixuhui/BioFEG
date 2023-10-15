
import os
import argparse
import pickle
import torch
import json
import sys
import io
import random
import time
import numpy as np

from multiprocessing.pool import ThreadPool

from tqdm import tqdm, trange
from collections import OrderedDict

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_transformers.optimization import WarmupLinearSchedule
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.modeling_utils import WEIGHTS_NAME

from blink.biencoder.biencoder import BiEncoderRanker, load_biencoder,ConditionalGenerator, ConditionalDiscriminator, DiscriptionEmbedding, calc_gradient_penalty
import logging

import blink.candidate_ranking.utils as utils
import blink.biencoder.data_process as data
from blink.biencoder.zeshel_utils import DOC_PATH, WORLDS, world_to_id
from blink.common.optimizer import get_bert_optimizer, get_bert_parameters, get_other_parameters
from blink.common.params import BlinkParser
from pytorch_transformers.optimization import AdamW
from torch.autograd import Variable


logger = None

# The evaluate function during training uses in-batch negatives:
# for a batch of size B, the labels from the batch are used as label candidates
# B is controlled by the parameter eval_batch_size
def evaluate(
    reranker, eval_dataloader, params, device, logger, generator, entity_embedding,
):
    with torch.set_grad_enabled(False):
        reranker.model.eval()
        generator.eval()
        entity_embedding.eval()
        if params["silent"]:
            iter_ = eval_dataloader
        else:
            iter_ = tqdm(eval_dataloader, desc="Evaluation")

        results = {}
        eval_accuracy = 0.0
        nb_eval_examples = 0
        nb_eval_steps = 0

        for step, batch in enumerate(iter_):
            batch = tuple(t.to(device) for t in batch)
            if params["zeshel"]:
                context_input, candidate_input, label_input, _ = batch
            else:
                context_input, candidate_input, label_input = batch

            gen_feats = generate(generator, entity_embedding, candidate_input)
            entities_real_embedding = reranker.encode_candidate(candidate_input).cuda(device)
            scores = torch.matmul(gen_feats, entities_real_embedding.t())
            logits = scores.detach().cpu().numpy()
            # Using in-batch negatives, the label ids are diagonal
            label_ids = torch.LongTensor(
                torch.arange(params["eval_batch_size"])
            ).numpy()
            tmp_eval_accuracy, _ = utils.accuracy(logits, label_ids)

            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += context_input.size(0)
            nb_eval_steps += 1

        normalized_eval_accuracy = eval_accuracy / nb_eval_examples
        logger.info("Eval accuracy: %.5f" % normalized_eval_accuracy)
        results["normalized_accuracy"] = normalized_eval_accuracy
        return results


def generate(generator, entity_embedding, candidate_input):
    entities = entity_embedding(candidate_input)
    entities_v = Variable(entities)
    bs = candidate_input.size(0)
    noises = torch.randn(bs, 768).cuda(candidate_input.device)
    noises = Variable(noises)
    feats = generator(noises, entities_v)
    return feats


def main(params):
    model_output_path = params["output_path"]
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    logger = utils.get_logger(params["output_path"])

    # Init model
    reranker = BiEncoderRanker(params)
    tokenizer = reranker.tokenizer
    reranker.model.eval()
    device = reranker.device

    generator = ConditionalGenerator()
    discriminator = ConditionalDiscriminator()
    entity_embedding = DiscriptionEmbedding(params)
    generator.cuda(device)
    discriminator.cuda(device)
    entity_embedding.cuda(device)


    if params["gradient_accumulation_steps"] < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                params["gradient_accumulation_steps"]
            )
        )

    # An effective batch size of `x`, when we are accumulating the gradient accross `y` batches will be achieved by having a batch size of `z = x / y`
    # args.gradient_accumulation_steps = args.gradient_accumulation_steps // n_gpu
    params["train_batch_size"] = (
        params["train_batch_size"] // params["gradient_accumulation_steps"]
    )
    train_batch_size = params["train_batch_size"]
    eval_batch_size = params["eval_batch_size"]
    grad_acc_steps = params["gradient_accumulation_steps"]

    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load train data
    train_samples = utils.read_dataset("train", params["data_path"])
    logger.info("Read %d train samples." % len(train_samples))

    train_data, train_tensor_data = data.process_mention_data(
        train_samples,
        tokenizer,
        params["max_context_length"],
        params["max_cand_length"],
        context_key=params["context_key"],
        silent=params["silent"],
        logger=logger,
        debug=params["debug"],
    )
    if params["shuffle"]:
        train_sampler = RandomSampler(train_tensor_data)
    else:
        train_sampler = SequentialSampler(train_tensor_data)

    train_dataloader = DataLoader(
        train_tensor_data, sampler=train_sampler, batch_size=train_batch_size
    )

    # Load eval data
    # TODO: reduce duplicated code here
    valid_samples = utils.read_dataset("val", params["data_path"])
    logger.info("Read %d valid samples." % len(valid_samples))

    valid_data, valid_tensor_data = data.process_mention_data(
        valid_samples,
        tokenizer,
        params["max_context_length"],
        params["max_cand_length"],
        context_key=params["context_key"],
        silent=params["silent"],
        logger=logger,
        debug=params["debug"],
    )
    valid_sampler = SequentialSampler(valid_tensor_data)
    valid_dataloader = DataLoader(
        valid_tensor_data, sampler=valid_sampler, batch_size=eval_batch_size
    )

    # evaluate before training
    results = evaluate(
        reranker, valid_dataloader, params, device=device, logger=logger, generator=generator, entity_embedding=entity_embedding,
    )

    number_of_samples_per_dataset = {}

    time_start = time.time()

    utils.write_to_file(
        os.path.join(model_output_path, "training_params.txt"), str(params)
    )

    logger.info("Starting training")
    description_params = get_bert_parameters([entity_embedding], params["type_optimization"])
    generator_params = get_other_parameters([generator])
    discriminator_params = get_other_parameters([discriminator])
    d_params = discriminator_params + description_params

    optimizer_d = AdamW(
        d_params,
        lr=params["learning_rate"],
        correct_bias=False
    )
    optimizer_g = AdamW(
        generator_params,
        lr=params["learning_rate"],
        correct_bias=False
    )
    one = torch.ones([]).to(device)
    mone = one * -1


    best_epoch_idx = -1
    best_score = -1

    num_train_epochs = params["num_train_epochs"]
    for epoch_idx in trange(int(num_train_epochs), desc="Epoch"):
        train_g_losses = []
        train_d_losses = []
        train_r_losses = []

        if params["silent"]:
            iter_ = train_dataloader
        else:
            iter_ = tqdm(train_dataloader, desc="Batch")

        with torch.set_grad_enabled(True):
            reranker.model.eval()
            discriminator.train()
            generator.train()
            entity_embedding.train()

            for p in reranker.model.parameters():
                p.requires_grad = False

            for step, batch in enumerate(iter_):

                for p in discriminator.parameters():
                    p.requires_grad = True

                batch = tuple(t.to(device) for t in batch)
                if params["zeshel"]:
                    context_input, candidate_input, label_input, _ = batch
                else:
                    context_input, candidate_input, label_input = batch

                for it_c in range(params["critic_iters"]):
                    real_feats = reranker.encode_context(context_input).cuda(device)

                    optimizer_d.zero_grad()

                    entities = entity_embedding(candidate_input)
                    real_feats_v = Variable(real_feats)
                    entities_v = Variable(entities)

                    real_logits = discriminator(real_feats_v, entities_v)
                    critic_d_real = real_logits.mean()
                    critic_d_real.backward(mone, retain_graph=False)

                    fake_feats = generate(generator, entity_embedding, candidate_input)
                    fake_feats = torch.relu(fake_feats)

                    fake_logits = discriminator(fake_feats.detach(), entities_v)
                    critic_d_fake = fake_logits.mean()
                    critic_d_fake.backward(one, retain_graph=False)

                    gp = calc_gradient_penalty(discriminator, real_feats, fake_feats.data, entities)
                    gp.backward()

                    d_cost = critic_d_fake - critic_d_real + gp
                    train_d_losses.append(d_cost.data.cpu().numpy())
                    optimizer_d.step()

                for p in discriminator.parameters():
                    p.requires_grad = False

                real_feats = reranker.encode_context(context_input).cuda(device)

                optimizer_g.zero_grad()

                entities = entity_embedding(candidate_input)
                entities_v = Variable(entities)

                fake_feats = generate(generator, entity_embedding, candidate_input)
                fake_feats = torch.relu(fake_feats)

                recon_loss = torch.nn.functional.mse_loss(fake_feats, real_feats, reduction='mean')
                train_r_losses.append(recon_loss.data.cpu().numpy())
                fake_logits = discriminator(fake_feats, entities_v)
                critic_g_fake = fake_logits.mean()
                g_cost = -critic_g_fake
                train_g_losses.append(g_cost.data.cpu().numpy())
                g_cost.backward()
                optimizer_g.step()

        logger.info(
            "disc loss {} - gen loss {} mse loss: {}\n".format(
                np.mean(train_d_losses),
                np.mean(train_g_losses),
                np.mean(train_r_losses),
            )
        )

        results = evaluate(
            reranker, valid_dataloader, params, device=device, logger=logger, generator=generator, entity_embedding=entity_embedding,
        )

        start_saving = 20
        save_every = 10
        if (epoch_idx + 1) % save_every == 0 and epoch_idx + 1 >= start_saving:
            logger.info("***** Saving fine - tuned model *****")
            gan_model = {'generator': generator.state_dict(),
                         'discriminator': discriminator.state_dict(),
                         'entity_embedding': entity_embedding.state_dict()}
            epoch_output_folder_path = os.path.join(
                model_output_path, "epoch_{}".format(epoch_idx)
            )
            torch.save(gan_model, epoch_output_folder_path)

    execution_time = (time.time() - time_start) / 60
    utils.write_to_file(
        os.path.join(model_output_path, "training_time.txt"),
        "The training took {} minutes\n".format(execution_time),
    )
    logger.info("The training took {} minutes\n".format(execution_time))




if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_training_args()
    parser.add_eval_args()

    # args = argparse.Namespace(**params)
    args = parser.parse_args()
    print(args)

    params = args.__dict__
    main(params)
