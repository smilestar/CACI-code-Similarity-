import torch
from torch_geometric.data import Dataset, Batch, Data
import json
from torch.utils.data import Dataset, DataLoader
import torch
import random
import json
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
import argparse
import sys
import traceback

class PairData(Data):
    def __init__(self, edge_index_a, x_a, edge_index_b, x_b, sim_y):
        super(PairData, self).__init__()
        self.edge_index_a = edge_index_a
        self.x_a = x_a
        self.edge_index_b = edge_index_b
        self.x_b = x_b
        self.sim_y = sim_y

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == 'edge_index_a':
            return self.x_a.size(0)
        if key == 'edge_index_b':
            return self.x_b.size(0)
        return super().__inc__(key, value, *args, **kwargs)

class BERTDataset(Dataset):
    def __init__(self, cross_corpus_path, nsp_corpus_path, vocab, seq_len, encoding="utf-8", on_memory=True):
        self.vocab = vocab
        self.seq_len = seq_len
        self.on_memory = on_memory
        self.corpus_lines = []

        with open(cross_corpus_path, "r", encoding=encoding) as f1:
            self.cross_data = json.load(f1)
        with open(nsp_corpus_path, "r", encoding=encoding) as f2:
            self.nsp_data = json.load(f2)

        self.corpus_lines = min(len(self.cross_data["1"]), len(self.nsp_data["1"]))

    def __len__(self):
        return self.corpus_lines


    def __getitem__(self, index):
        aligned_a, aligned_b, aligned_label, n1, n2, n_label = self.random_sent(index)

        n1_random, n1_label = self.random_word(n1)
        n2_random, n2_label = self.random_word(n2)

        n1 = [self.vocab.sos_index] + n1_random + [self.vocab.eos_index]
        n2 = n2_random + [self.vocab.eos_index]

        aligned_a = [self.vocab.sos_index] + [self.vocab.stoi.get(token, self.vocab.unk_index) for token in aligned_a.split()] + [self.vocab.eos_index]
        aligned_b = [self.vocab.sos_index] + [self.vocab.stoi.get(token, self.vocab.unk_index) for token in aligned_b.split()] + [self.vocab.eos_index]

        n1_label = [self.vocab.pad_index] + n1_label + [self.vocab.pad_index]
        n2_label = n2_label + [self.vocab.pad_index]

        nsp_segment_label = ([1 for _ in range(len(n1))] + [2 for _ in range(len(n2))])[:self.seq_len]
        aligned_a_segment_label = [1 for _ in range(len(aligned_a))][:self.seq_len]
        aligned_b_segment_label = [2 for _ in range(len(aligned_b))][:self.seq_len]

        # 生成输入数据
        nsp_bert_input = (n1 + n2)[:self.seq_len]
        nsp_bert_label = (n1_label + n2_label)[:self.seq_len]
        aligned_a_bert_input = aligned_a[:self.seq_len]
        aligned_b_bert_input = aligned_b[:self.seq_len]

        # 填充数据，确保序列长度一致
        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(nsp_bert_input))]
        nsp_bert_input.extend(padding), nsp_bert_label.extend(padding), nsp_segment_label.extend(padding)

        aligned_a_padding = [self.vocab.pad_index for _ in range(self.seq_len - len(aligned_a_bert_input))]
        aligned_a_bert_input.extend(aligned_a_padding), aligned_a_segment_label.extend(aligned_a_padding)

        aligned_b_padding = [self.vocab.pad_index for _ in range(self.seq_len - len(aligned_b_bert_input))]
        aligned_b_bert_input.extend(aligned_b_padding), aligned_b_segment_label.extend(aligned_b_padding)

        output = {
            "nsp_bert_input": nsp_bert_input,
            "nsp_bert_label": nsp_bert_label,
            "nsp_segment_label": nsp_segment_label,
            "nsp_is_next": n_label,
            "aligned_a_bert_input": aligned_a_bert_input,
            "aligned_a_segment_label": aligned_a_segment_label,
            "aligned_b_bert_input": aligned_b_bert_input,
            "aligned_b_segment_label": aligned_b_segment_label,
            "aligned_label": aligned_label
        }

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index

                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(0)

        return tokens, output_label


    def random_sent(self, index):
        cross_item, nsp_item = self.get_corpus_item(index)
        return (cross_item['instruction_a'], cross_item['instruction_b'], cross_item['label'],
                nsp_item['sentence_a'], nsp_item['sentence_b'], nsp_item['label'])


    def get_corpus_item(self, item):
        if self.on_memory:
            rand = random.random()
            if rand < 0.25:
                return self.cross_data["0"][item], self.nsp_data["0"][item]
            elif rand < 0.5:
                return self.cross_data["0"][item], self.nsp_data["1"][item]
            elif rand < 0.75:
                return self.cross_data["1"][item], self.nsp_data["0"][item]
            else:
                return self.cross_data["1"][item], self.nsp_data["1"][item]
            

        return len(self.train_dataset)

    @property
    def raw_file_names(self):
        return None

    @property
    def processed_file_names(self):
        return None

    def __len__(self):
        return len(self.train_dataset)

    def get(self, idx):
        data = self.train_dataset[idx]
        x_a, edge_index_a, num_nodes_a = self.create_graph_data(data['graph_a'])
        x_b, edge_index_b, num_nodes_b = self.create_graph_data(data['graph_b'])
        label = torch.tensor([data['label']], dtype=torch.float)

        graph_data = PairData(
            x_a=x_a,
            edge_index_a=edge_index_a,
            x_b=x_b,
            edge_index_b=edge_index_b,
            sim_y=label
        )
        return graph_data

    def __getitem__(self, idx):
        try:
            data = self.get(idx)
            if data is None:
                raise ValueError(f"Data at index {idx} is None")
            return data
        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            raise

    @staticmethod
    def create_graph_data(graph):
        nodes = torch.tensor(graph['features'], dtype=torch.float)
        edges = torch.tensor(graph['edges'], dtype=torch.long)

        # 为无边图添加自环边
        if edges.numel() == 0:
            edges = torch.tensor([[0, 0]], dtype=torch.long)

        num_nodes = nodes.size(0)  # 节点数为特征矩阵的行数
        return nodes, edges.t(), num_nodes  # 边索引需要转置

class GATModel(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, emb_size=64, num_heads=8, time_steps=5, dropout=0.1):
        super(GATModel, self).__init__()
        self.time_steps = time_steps
        self.lin = torch.nn.Linear(in_channels, emb_size, bias=False)
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True, dropout=dropout)
        self.conv3 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True, dropout=dropout)
        self.conv4 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True, dropout=dropout)
        self.conv5 = GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False)

    def readout(self, h0, hn):
        h_cat = torch.cat((h0, hn), dim=0)
        res = self.lin(h_cat)
        emb = torch.sum(res, dim=0)
        return emb

    def forward(self, h0, edge_index):
        hn = h0
        hn = self.conv1(hn, edge_index)
        hn = F.relu(hn)
        hn = self.conv2(hn, edge_index)
        hn = F.relu(hn)
        hn = self.conv3(hn, edge_index)
        hn = F.relu(hn)
        hn = self.conv4(hn, edge_index)
        hn = F.relu(hn)
        hn = self.conv5(hn, edge_index)
        hn = F.relu(hn)
        res = self.readout(h0, hn)
        return res

class BERTLM(nn.Module):

    def __init__(self, bert: BERT, vocab_size):
        super().__init__()
        self.bert = bert
        self.NSP = NextSentencePrediction(self.bert.hidden)
        self.MLM = MaskedLanguageModel(self.bert.hidden, vocab_size)

    def forward(self, nsp_bert_input, nsp_segment_label,
                aligned_a_bert_input, aligned_a_segment_label,
                aligned_b_bert_input, aligned_b_segment_label):

        nsp_output = self.bert(nsp_bert_input, nsp_segment_label)
        aligned_a_output = self.bert(aligned_a_bert_input, aligned_a_segment_label)
        aligned_b_output = self.bert(aligned_b_bert_input, aligned_b_segment_label)

        return self.NSP(nsp_output), self.MLM(nsp_output), aligned_a_output, aligned_b_output

    def compute_contrastive_loss(self, aligned_a_output, aligned_b_output):
        aligned_a_output_norm = F.normalize(aligned_a_output[:, 0], p=2, dim=-1)  
        aligned_b_output_norm = F.normalize(aligned_b_output[:, 0], p=2, dim=-1)  

        cosine_sim = torch.sum(aligned_a_output_norm * aligned_b_output_norm, dim=-1)

        margin = 0.5
        contrastive_loss = torch.mean(
            (1 - cosine_sim) ** 2 + F.relu(cosine_sim - margin) ** 2
        )

        return contrastive_loss


class NextSentencePrediction(nn.Module):

    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))


class MaskedLanguageModel(nn.Module):

    def __init__(self, hidden, vocab_size):
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))
    
class BERT(nn.Module):

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        self.feed_forward_hidden = hidden * 4

        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x, segment_info):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        x = self.embedding(x, segment_info)

        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x

    def encode(self, x, segment_info):

        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        x = self.embedding(x, segment_info)

        for transformer in self.transformer_blocks[:-1]:
            x = transformer.forward(x, mask)

        return x
    
class BERTTrainer:
    """
    BERTTrainer用于预训练BERT模型，包含以下训练任务：
        1. Masked Language Model (MLM)
        2. Next Sentence Prediction (NSP)
        3. Cross-architecture Alignment Task (for aligning instruction pairs across architectures)
    """

    def __init__(self, bert: BERT, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10):
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        self.bert = bert
        self.model = BERTLM(bert, vocab_size).to(self.device)

        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        self.train_data = train_dataloader
        self.test_data = test_dataloader

        self.optim = AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)

        self.masked_criterion = nn.NLLLoss(ignore_index=0)
        self.nsp_criterion = nn.NLLLoss()
        self.log_freq = log_freq

        self.loss_history = {
            "MLM": [],
            "NSP": [],
            "CrossArch": [],
            "TOTAL": []
        }

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        str_code = "train" if train else "test"

        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}

            nsp_output, mask_lm_output, aligned_a_output, aligned_b_output = self.model.forward(
                data["nsp_bert_input"], data["nsp_segment_label"],
                data["aligned_a_bert_input"], data["aligned_a_segment_label"],
                data["aligned_b_bert_input"], data["aligned_b_segment_label"]
            )

            nsp_loss = self.nsp_criterion(nsp_output, data["nsp_is_next"])

            mask_loss = self.masked_criterion(mask_lm_output.transpose(1, 2), data["nsp_bert_label"])

            cross_arch_loss = self.contrastive_loss(aligned_a_output[:, 0], aligned_b_output[:, 0], data["aligned_label"])

            total_loss = nsp_loss + mask_loss + cross_arch_loss

            if train:
                self.optim_schedule.zero_grad()
                total_loss.backward()
                self.optim_schedule.step_and_update_lr()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "MLM": mask_loss.item(),
                "NSP": nsp_loss.item(),
                "CrossArch": cross_arch_loss.item(),
                "TOTAL": total_loss.item(),
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
                self.loss_history["MLM"].append(mask_loss.item())
                self.loss_history["NSP"].append(nsp_loss.item())
                self.loss_history["CrossArch"].append(cross_arch_loss.item())
                self.loss_history["TOTAL"].append(total_loss.item())

    def contrastive_loss(self, em1, em2, label, margin=5.0):
        # 计算欧式距离
        distance = torch.norm(em1 - em2, p=2, dim=1)

        # 计算对比损失
        loss_pos = label * 0.5 * torch.pow(distance, 2)
        loss_neg = (1 - label) * 0.5 * torch.pow(F.relu(margin - distance), 2)

        # 返回平均损失
        return torch.mean(loss_pos + loss_neg)

    def save(self, epoch, file_path="output/bert_trained.model"):
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.bert.cpu(), output_path)
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path


def train_bert():
    vocab_path = "dataset/vocab"
    train_nsp_path = "dataset/nsp_data_train.json"
    train_cross_path = "dataset/cross_arch_contrastive_data_train.json"
    test_nsp_path = "dataset/nsp_data_test.json"
    test_cross_path = "dataset/cross_arch_contrastive_data_test.json"
    output_path = "output/bert_model"

    vocab = load_vocab(vocab_path)

    print("Loading Training Dataset...")
    train_dataset = BERTDataset(train_cross_path, train_nsp_path, vocab, seq_len=20, on_memory=True)

    print("Creating Training Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=256, num_workers=25, persistent_workers=True)

    print("Loading Testing Dataset...")
    test_dataset = BERTDataset(test_cross_path, test_nsp_path, vocab, seq_len=20, on_memory=True)

    print("Creating Testing Dataloader")
    test_data_loader = DataLoader(test_dataset, batch_size=256, num_workers=25, persistent_workers=True)

    print("Building BERT model")
    bert = BERT(len(vocab), hidden=32, n_layers=12, attn_heads=8, dropout=0.1)

    print("Creating BERT Trainer")
    trainer = BERTTrainer(bert, len(vocab), train_dataloader=train_data_loader,
                                  test_dataloader=test_data_loader,
                                  lr=1e-5, betas=(0.9, 0.999), weight_decay=0.01,
                                  with_cuda=True, cuda_devices=[0, 1], log_freq=100)


    print("Training Start")
    for epoch in range(5):
        trainer.train(epoch)
        trainer.save(epoch, output_path)
        trainer.plot_loss(epoch)
        if test_data_loader is not None:
            trainer.test(epoch)
            
class GraphPairDataset(Dataset):

    def __init__(self, train_dataset_path):
        super().__init__()

        self.train_dataset_path = train_dataset_path

        with open(train_dataset_path, 'r') as f:
            self.train_dataset = json.load(f)


    def __len__(self):
        return len(self.train_dataset)

    @property
    def raw_file_names(self):
        return None

    @property
    def processed_file_names(self):
        return None

    def __len__(self):
        return len(self.train_dataset)

    def get(self, idx):
        data = self.train_dataset[idx]
        x_a, edge_index_a, num_nodes_a = self.create_graph_data(data['graph_a'])
        x_b, edge_index_b, num_nodes_b = self.create_graph_data(data['graph_b'])
        label = torch.tensor([data['label']], dtype=torch.float)

        graph_data = PairData(
            x_a=x_a,
            edge_index_a=edge_index_a,
            x_b=x_b,
            edge_index_b=edge_index_b,
            sim_y=label
        )
        return graph_data

    def __getitem__(self, idx):
        try:
            data = self.get(idx)
            if data is None:
                raise ValueError(f"Data at index {idx} is None")
            return data
        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            raise

    @staticmethod
    def create_graph_data(graph):
        nodes = torch.tensor(graph['features'], dtype=torch.float)
        edges = torch.tensor(graph['edges'], dtype=torch.long)

        # 为无边图添加自环边
        if edges.numel() == 0:
            edges = torch.tensor([[0, 0]], dtype=torch.long)

        num_nodes = nodes.size(0)  # 节点数为特征矩阵的行数
        return nodes, edges.t(), num_nodes  # 边索引需要转置

class GATTrainer:
    """
    GATTrainer用于预训练GAT模型，包含以下训练任务：
        1. 节点分类任务
        2. 边预测任务
        3. 图级别的特定任务（可扩展）
    """

    def __init__(self, gat: GATModel, train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-3, weight_decay: float = 5e-4, with_cuda: bool = True,
                 cuda_devices=None, log_freq: int = 10):
        """
        :param gat: GAT模型
        :param train_dataloader: 训练数据
        :param test_dataloader: 测试数据（可选）
        :param lr: 学习率
        :param weight_decay: 权重衰减
        :param with_cuda: 是否使用CUDA
        :param log_freq: 日志频率
        """
        self.device = torch.device("cuda:1" if torch.cuda.is_available() and with_cuda else "cpu")
        self.gat = gat.to(self.device)

        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUs for GAT" % torch.cuda.device_count())
            self.gat = nn.DataParallel(self.gat, device_ids=cuda_devices)

        self.train_data = train_dataloader
        self.test_data = test_dataloader

        self.optim = torch.optim.Adam(self.gat.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = ContrastiveLoss()

        self.log_freq = log_freq
        self.loss_history = {"Train": [], "Test": []}
        print("Total Parameters:", sum(p.numel() for p in self.gat.parameters()))

    def train(self, epoch):
        self.iteration(epoch, self.train_data, train=True)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        str_code = "train" if train else "test"
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        total_loss = 0
        for i, data in data_iter:
            data = data.to(self.device)

            embedding_a = self.gat(data.x_a, data.edge_index_a)
            embedding_b = self.gat(data.x_b, data.edge_index_b)
            sim_label = data.sim_y

            loss = self.criterion(embedding_a.unsqueeze(0), embedding_b.unsqueeze(0), sim_label)

            if train:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "loss": loss.item()
            }

            total_loss += loss.item()
            if i % self.log_freq == 0:
                # data_iter.set_postfix({"loss": loss.item()})
                data_iter.write(str(post_fix))

        avg_loss = total_loss / len(data_loader)
        self.loss_history["Train" if train else "Test"].append(avg_loss)
        print(f"EP_{str_code}:{epoch} Avg Loss: {avg_loss:.4f}")

    def save(self, epoch, file_path="output/gat_trained.model"):
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.gat.cpu(), output_path)
        self.gat.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path

    def plot_loss(self, epoch, save_file="img/gat_training_loss"):
        plt.figure(figsize=(10, 6))
        for key, values in self.loss_history.items():
            plt.plot(values, label=f"{key} Loss")
        plt.title("GAT Training Loss Curves")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        output_path = save_file + ".ep%d" % epoch + ".png"
        plt.savefig(output_path)
        print("Loss Curves Saved on:", save_file + ".png")

    @staticmethod
    def contrastive_loss(h1, h2, label, margin=10.0):
        """
        对比损失函数
        :param h1: 张量 (N, D)，图1的嵌入
        :param h2: 张量 (N, D)，图2的嵌入
        :param label: 张量 (N,)，表示样本对标签，1为正样本，0为负样本
        :param margin: 距离阈值
        :return: 损失值
        """
        # 计算嵌入向量的欧氏距离
        distance = torch.norm(h1 - h2, p=2, dim=1)

        # 正样本对的损失
        positive_loss = label * 0.5 * torch.pow(distance, 2)

        # 负样本对的损失
        negative_loss = (1 - label) * 0.5 * torch.pow(F.relu(margin - distance), 2)

        # 总损失
        loss = torch.mean(positive_loss + negative_loss)
        return loss

    def get_sim_percentage(self, distance, alpha=-0.1):
        # 确保 distance 是 CPU 张量
        if distance.is_cuda:  # 如果张量在 GPU 上
            distance = distance.cpu()  # 转移到 CPU
        similarity = 1 * np.exp(alpha * distance.detach().numpy())  # 转换为 NumPy 数组后计算
        return similarity

    def get_calc_label(self, eb1, eb2, threshold=0.7, alpha=-0.1):
        # 计算欧氏距离
        distance = torch.norm(eb1 - eb2, p=2, dim=1)
        # 获取相似度百分比
        percentage = self.get_sim_percentage(distance, alpha)
        # 根据阈值返回标签
        return (percentage >= threshold).astype(int)

    def calc_loss(self, eb1, eb2, labels, margin=0.40):
        euclidean_distance = torch.sum((eb1 - eb2)**2, dim=-1)
        return torch.mean(torch.nn.functional.relu(margin - labels * (1 - euclidean_distance)))

def train_gat():
    train_dataset_path = "dataset/gat_data.json"
    test_dataset_path = "dataset/test_gat_data.json"
    output_path = "output/gat_model"

    print("Loading Training Dataset...")
    train_dataset = GraphPairDataset(train_dataset_path)

    print("Creating Training Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=1, num_workers=15, shuffle=True, follow_batch=['x_a', 'x_b'])

    print("Loading Testing Dataset...")
    test_dataset = GraphPairDataset(test_dataset_path)

    print("Creating Testing Dataloader")
    test_data_loader = DataLoader(test_dataset, batch_size=1, num_workers=15, shuffle=False, follow_batch=['x_a', 'x_b'])

    # 定义模型
    print("Building GAT model")
    gat = GATModel(in_channels=32, hidden_channels=128, out_channels=32, dropout=0.0)

    print("Creating GAT Trainer")
    trainer = GATTrainer(gat=gat, train_dataloader=train_data_loader,
               test_dataloader=test_data_loader,
               lr=1e-4, with_cuda=True, cuda_devices=[1], log_freq=200, weight_decay=0.01)

    print("Training Start")
    for epoch in range(75):
        trainer.train(epoch)
        trainer.save(epoch, output_path)
        trainer.plot_loss(epoch)
        if test_data_loader is not None:
            trainer.test(epoch)
            
def main():

    parser = argparse.ArgumentParser(description="Training Framework")

    # 通用参数
    parser.add_argument("--mode", type=str, choices=["bert", "gat", "both"], required=True,
                        help="Choose training mode: 'bert', 'gat', or 'both'")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs (default: 10)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for dataloader (default: 128)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--cuda", action="store_true",
                        help="Use CUDA if available")
    parser.add_argument("--devices", type=int, nargs="+", default=[0],
                        help="CUDA device ids, e.g. --devices 0 1 (default: [0])")

    args = parser.parse_args()

    try:
        if args.mode == "bert":
            print(f"Starting BERT training for {args.epochs} epochs...")
            train_bert()  # 可以在内部改造传入 epochs, lr, batch_size
        elif args.mode == "gat":
            print(f"Starting GAT training for {args.epochs} epochs...")
            train_gat()
        elif args.mode == "both":
            print(f"Starting BOTH training: BERT ({args.epochs} epochs) then GAT ({args.epochs} epochs)...")
            train_bert()
            train_gat()
        else:
            raise ValueError("Invalid mode. Please choose 'bert', 'gat' or 'both'.")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl+C). Exiting...")
        sys.exit(1)
    except Exception as e:
        print("An error occurred during training:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
    