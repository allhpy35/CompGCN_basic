from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from ordered_set import OrderedSet
import torch

REL2CLS = {
    'circ-mirna': 0,
    'mirna-disease': 1,
    'mirna-lncrna': 2,
    'lncrna-disease': 3,
    'circ-disease': 4
}


class KGRACDAClsDataset(Dataset):
    def __init__(self, data):
        self.data = data  # [(sub, obj, y)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def collate_fn(batch):
        sub, obj, y = zip(*batch)
        return (
            torch.LongTensor(sub),
            torch.LongTensor(obj),
            torch.LongTensor(y)
        )


class KGRACDAClsLoader:
    def __init__(self, params):
        self.p = params
        self.load_data()
        self.edge_index, self.edge_type = self.construct_adj()
        # entity 수
        self.p.num_ent = len(self.ent2id)

        # relation 수 (CLS에서는 class 수)
        self.p.num_rel = self.p.num_classes

        # embedding dim 보장
        if self.p.embed_dim is None:
            self.p.embed_dim = self.p.gcn_dim

    def load_data(self):
        ent_set = OrderedSet()
        rel_set = OrderedSet()

        raw_data = defaultdict(list)

        for split in ['train', 'valid', 'test']:
            #with open(f'./data/{self.p.dataset}/{self.p.data_name}/{split}.txt') as f:
            with open(f'./data/{self.p.dataset}/dataset2/og_data/{split}.txt') as f:
                for line in f:
                    sub, rel, obj = map(str.lower, line.strip().split('\t'))
                    if rel not in REL2CLS:
                        continue

                    ent_set.add(sub)
                    ent_set.add(obj)
                    rel_set.add(rel)

                    raw_data[split].append((sub, obj, REL2CLS[rel]))

        # entity mapping
        self.ent2id = {e: i for i, e in enumerate(ent_set)}
        self.id2ent = {i: e for e, i in self.ent2id.items()}
        self.p.num_ent = len(self.ent2id)
        self.p.num_rel = len(REL2CLS)

        # convert to id
        self.data = {}
        for split in raw_data:
            self.data[split] = [
                (self.ent2id[s], self.ent2id[o], y)
                for s, o, y in raw_data[split]
            ]

        # dataloader
        self.data_iter = {
            split: DataLoader(
                KGRACDAClsDataset(self.data[split]),
                batch_size=self.p.batch_size,
                shuffle=(split == 'train'),
                collate_fn=KGRACDAClsDataset.collate_fn
            )
            for split in ['train', 'valid', 'test']
        }

    def construct_adj(self):
        """
        GCN encoder용 adjacency
        모든 (sub, rel, obj)를 edge로 사용
        """
        edge_index = []
        edge_type = []

        for split in self.data:
            for s, o, y in self.data[split]:
                edge_index.append([s, o])
                edge_type.append(y)

        edge_index = torch.LongTensor(edge_index).t().contiguous()
        edge_type = torch.LongTensor(edge_type)
        return edge_index, edge_type
