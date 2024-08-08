import argparse
import torch
from Self_Supervised_Representation_Augmentor.SEComm import (
    Encoder,
    Model,
    drop_feature,
    SelfExpr,
    ClusterModel,
    MergeLayer,
)
from Self_Supervised_Representation_Augmentor.FeatureExtractor import FeatureExtractor
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)
from torch_geometric.loader import TemporalDataLoader


class Args:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.data_size = 0
        self.device = "cuda:0"
        self.num_nodes = 0
        self.args = None

    def get_args(self):
        self.parser.add_argument("--gpu_id", type=str, default="0")
        self.parser.add_argument("--drop_feature_rate_1", type=float, default=0.0)
        self.parser.add_argument("--drop_feature_rate_2", type=float, default=0.05)
        self.parser.add_argument("--threshold", type=float, default=0.5)
        self.parser.add_argument("--batch_size", type=int, default=128)
        self.parser.add_argument("--norm", type=bool, default=True)
        self.parser.add_argument("--PRETRAIN_THRESHOLD", type=float, default=0.5)
        self.parser.add_argument("--SE_THRESHOLD", type=float, default=0.8)
        self.parser.add_argument("--EMBEDDING_DIM", type=int, default=128)
        self.parser.add_argument("--embedding_dim", type=int, default=128)
        self.parser.add_argument("--memory_dim", type=int, default=128)
        self.parser.add_argument("--time_dim", type=int, default=128)
        self.parser.add_argument("--iterations", type=int, default=1)
        self.parser.add_argument("--n_class", type=int, default=2)
        self.parser.add_argument("--num_cl_hidden", type=int, default=64)
        self.parser.add_argument("--drop_out", type=float, default=0.1)
        self.parser.add_argument("--token", type=str, default="BoT_test")
        self.parser.add_argument(
            "--dataroot",
            type=str,
            default="/new_disk/3D-IDS-Euler/UNIDS/data/result/CIC-ToN-IoT.pt",
        )
        self.parser.add_argument("--pretrain_lr", type=float, default=0.0005)
        self.parser.add_argument("--se_lr", type=float, default=0.0005)
        self.parser.add_argument("--full_lr", type=float, default=0.0001)
        self.parser.add_argument("--weight_decay", type=float, default=0.00001)
        self.parser.add_argument("--truncate", type=bool, default=False)
        self.parser.add_argument("--truncate_length", type=int, default=11)
        self.parser.add_argument("--use_mlp", type=bool, default=False)
        self.parser.add_argument("--cpu", type=bool, default=True)
        self.args = self.parser.parse_args()
        return self.args

    def get_device(self):
        if self.args.cpu:
            self.device = "cpu"
        else:
            self.device = "cuda:" + self.args.gpu_id
        return self.device

    def get_dataset(self):
        data = torch.load(self.args.dataroot)
        self.num_nodes = data.num_nodes
        data.msg = data.msg.to(torch.float32)
        if self.args.truncate:
            data.msg = data.msg[:, :, : self.args.truncate_length]
        print(data.msg.shape)
        # [651659, 9, 11] -> [651659, 99]
        data.msg = data.msg.reshape(data.msg.size(0), -1)
        print(data.msg.shape)
        self.data_size = data.msg.size(-1)
        return data

    def get_loader(self, data):
        return TemporalDataLoader(data, batch_size=self.args.batch_size)


    def get_model(self):
        args = self.args
        device = self.device
        if self.args.use_mlp:
            feature_extractor = FeatureExtractor(
                self.data_size, args.EMBEDDING_DIM, 128
            ).to(device)
        else:
            feature_extractor = None
            print("No feature extractor")
        neighbor_loader = LastNeighborLoader(self.num_nodes, size=10, device=device)
        if self.args.use_mlp:
            memory = TGNMemory(
                self.num_nodes,
                args.EMBEDDING_DIM,
                args.memory_dim,
                args.time_dim,
                message_module=IdentityMessage(
                    args.EMBEDDING_DIM, args.memory_dim, args.time_dim
                ),
                aggregator_module=LastAggregator(),
            ).to(device)
        else:
            memory = TGNMemory(
                self.num_nodes,
                self.data_size,
                args.memory_dim,
                args.time_dim,
                message_module=IdentityMessage(
                    self.data_size, args.memory_dim, args.time_dim
                ),
                aggregator_module=LastAggregator(),
            ).to(device)
        mergelayer = MergeLayer(
            args.embedding_dim, args.embedding_dim, args.embedding_dim
        ).to(device)
        if self.args.use_mlp:
            optimizer = torch.optim.Adam(
                set(memory.parameters())
                | set(mergelayer.parameters())
                | set(feature_extractor.parameters()),
                lr=0.0005,
            )
        else:
            optimizer = torch.optim.Adam(
                set(memory.parameters()) | set(mergelayer.parameters()), lr=0.0005
            )
        semodel = SelfExpr(args.batch_size).to(device)
        seoptimizer = torch.optim.Adam(
            semodel.parameters(), lr=0.0005, weight_decay=0.00001
        )
        clustermodel = ClusterModel(
            args.embedding_dim * 2, args.num_cl_hidden, args.n_class, args.drop_out
        ).to(device)
        clusteroptimizer = torch.optim.Adam(
            clustermodel.parameters(), lr=args.se_lr, weight_decay=args.weight_decay
        )
        if self.args.use_mlp:
            fulloptimizer = torch.optim.Adam(
                set(memory.parameters())
                | set(mergelayer.parameters())
                | set(clustermodel.parameters())
                | set(feature_extractor.parameters()),
                lr=args.full_lr,
                weight_decay=args.weight_decay,
            )
        else:
            fulloptimizer = torch.optim.Adam(
                set(memory.parameters())
                | set(mergelayer.parameters())
                | set(clustermodel.parameters()),
                lr=args.full_lr,
                weight_decay=args.weight_decay,
            )
        return (
            feature_extractor,
            neighbor_loader,
            memory,
            mergelayer,
            optimizer,
            semodel,
            seoptimizer,
            clustermodel,
            clusteroptimizer,
            fulloptimizer,
        )

    def save_args(self, filename):
        with open(filename, "w") as f:
            for arg in vars(self.args):
                f.write(f"{arg}: {getattr(self.args, arg)}\n")
