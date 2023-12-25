import argparse
from exp.exp_FCDformer import exp_former

if __name__ == "__main__":
    # 建立解析对象
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='weather', help='data')
    parser.add_argument('--root_path', type=str, default='./data/public_data', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='weather.csv', help='data file')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')

    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of distillformer encoder')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--timeenc', type=int, default=0)
    parser.add_argument('--freq', type=str, default='h')

    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--checkpoints', type=str, default='/checkpoints', help='location of model checkpoints')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')

    parser.add_argument('--epoch', type=int, default=6, help='epoch_size')
    parser.add_argument('--fea_num', type=int, default=21, help='feature numbers except time_feature')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')

    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=4, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--fd_num', type=int, default=2, help='num of decompose sequence')

    parser.add_argument('--features', type=str, default='MS', help='target feature in S or MS task,S is all feature c_out=1,'
                                                                  'MS is only target, c_out=feature_num-1')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')


    args = parser.parse_args()
    print(args)

    Exp = exp_former
    exp = Exp(args)
    exp.train()
    exp.test()


