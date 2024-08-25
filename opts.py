import argparse
import torch

def parse_args():
    # 创建解析器
    parser = argparse.ArgumentParser(description='Training a model on a dataset')

    # 添加超参数和设置
    parser.add_argument('--lambda1', type=float, default=1, help='Weight for Lcaus loss')
    # parser.add_argument('--lambda2', type=float, default=1, help='Weight for Lr loss')
    parser.add_argument('--lambda3', type=float, default=1, help='Weight for Luni loss')
    parser.add_argument('--lambda4', type=float, default=0.5, help='Weight for Lxr loss')

    # 添加其他需要的设置
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--epochs_ft', type=int, default=150, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--supports', type=int, default=200, help='Supports number for training.')
    parser.add_argument('--lr_pre', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--lr_ft', type=float, default=0.0008, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 penalty).')
    parser.add_argument('--train_enc_dim', type=int, default=128, help='Hidden_dimension.')
    parser.add_argument('--lstm_layers', type=int, default=3, help='LSTM layers.')
    parser.add_argument('--num_heads', type=int, default=1, help='Number of heads.')
    parser.add_argument('--trans_layers', type=int, default=1, help='Transformers layers.')
    parser.add_argument('--embed_dim', type=int, default=1, help='Transformers embedding dimension.')
    parser.add_argument('--gcn_out_dim', type=int, default=16, help='GCN output dimension.')
    parser.add_argument('--causal_dim', type=int, default=128, help='Dimension of causal model.')

    # ... 你可以继续添加其他相关的设置

    # 解析参数
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # 根据args中的设置进行模型的训练和评估
    # ...

if __name__ == '__main__':
    main()
