

# Utility code for zeshel dataset
import json
import torch

DOC_PATH = "data/medmentions/documents/"

WORLDS = [
]

world_to_id = {src : k for k, src in enumerate(WORLDS)}


def load_entity_dict_zeshel(logger, params):
    entity_dict = {}
    # load data
    for i, src in enumerate(WORLDS):
        fname = DOC_PATH + src + ".json"
        cur_dict = {}
        doc_list = []
        src_id = world_to_id[src]
        with open(fname, 'rt') as f:
            for line in f:
                line = line.rstrip()
                item = json.loads(line)
                text = item["text"]
                doc_list.append(text[:256])

                if params["debug"]:
                    if len(doc_list) > 200:
                        break

        logger.info("Load for world %s." % src)
        entity_dict[src_id] = doc_list
    return entity_dict


class Stats():
    def __init__(self, top_k=1000):
        self.cnt = 0
        self.hits = []
        self.top_k = top_k
        self.rank = [1, 4, 8, 16, 32, 64, 100, 128, 256, 512]
        self.LEN = len(self.rank) 
        for i in range(self.LEN):
            self.hits.append(0)

    def add(self, idx):
        self.cnt += 1
        if idx == -1:
            return
        for i in range(self.LEN):
            if idx < self.rank[i]:
                self.hits[i] += 1

    def extend(self, stats):
        self.cnt += stats.cnt
        for i in range(self.LEN):
            self.hits[i] += stats.hits[i]

    def output(self):
        output_json = "Total: %d examples." % self.cnt
        for i in range(self.LEN):
            if self.top_k < self.rank[i]:
                break
            output_json += " r@%d: %.4f" % (self.rank[i], self.hits[i] / float(self.cnt))
        return output_json

