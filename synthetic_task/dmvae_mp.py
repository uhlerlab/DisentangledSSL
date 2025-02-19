from algorithms import *
from losses import *
from models import *
from dataset import *
from torch.utils.data import DataLoader
from utils import *
import warnings
warnings.filterwarnings("ignore")
import os
from argparse import ArgumentParser
import multiprocessing
import json
import random
from itertools import product

def init_lock(l):
    global lock
    lock = l

def set_seed(seed, use_cuda=True):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f'=> Seed of the run set to {seed}')


def wrapper_train_mp(args, a, seed, train_loader, test_loader, train_dataset, test_dataset):
    gpu_id = random.choice(args.gpus)  # Randomly choose from available GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)  # Set CUDA device only for this process
    args.seed = seed
    args.a = a
    set_seed(seed)
    logs = train_mp_dmvae(args, train_loader, test_loader, train_dataset, test_dataset)
    log_entry = {f'a_{a}_seed_{seed}': logs}
    lock.acquire()
    try:
        log_file_path = f"./results_synthetic_task_baseline/{args.data_mode}/dmvae_training_logs.json"
        with open(log_file_path, 'a') as file:  # Open file in append mode
            json.dump(log_entry, file, indent=4)
            file.write(",\n")  # Add comma and newline for JSON array format
    finally:
        lock.release()

    return log_entry


def train_mp_dmvae(args, train_loader, test_loader, train_dataset, test_dataset):
    dmvae = DMVAE(args.dim_info['X'], args.dim_info['Y'], args.hidden_dim, args.embed_dim, args.layers, args.activation, 
                        initialization = 'normal', a=args.a).cuda()
    joint_optim = optim.Adam(dmvae.parameters(), lr=args.lr)
    logs = dmvae.train_model_dmvae(train_loader, train_dataset, test_dataset, joint_optim, num_epoch=args.num_epoch, noise_scale=args.noise_scale, drop_scale=args.drop_scale)
    return logs


if __name__ == '__main__':
    parser = ArgumentParser()
    
    ## data params
    parser.add_argument('--data_mode', type=str, default='entangle')
    parser.add_argument('--num_data', type=int, default=90000)
    parser.add_argument('--dim_info', type=dict, default={'Y': 100, 'Z1': 50, 'Zs': 50, 'X': 100, 'Z2': 50})
    parser.add_argument('--weight_info', type=dict, default={'Zs': 50, 'Z1': 50, 'Z2': 50})
    parser.add_argument('--label_weight_info1', type=dict, default={'Zs': 0, 'Z1': 50, 'Z2': 0})
    parser.add_argument('--label_weight_info2', type=dict, default={'Zs': 50, 'Z1': 0, 'Z2': 0})
    parser.add_argument('--label_weight_info3', type=dict, default={'Zs': 0, 'Z1': 0, 'Z2': 50})
    parser.add_argument('--seed', type=int, default=0)
    
    # algorithm params
    parser.add_argument('--a', type=float, default=1e-5)
    parser.add_argument('--start_a', type=float, default=1e-5)

    ## model params
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--embed_dim', type=int, default=100)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    ## self-contrastive aug params
    parser.add_argument('--noise_scale', type=float, default=0.01)
    parser.add_argument('--drop_scale', type=float, default=10)

    # device
    parser.add_argument('--device', type=str, default="0", help="Comma-separated list of available GPUs (e.g., '0,1,2,3')")
    args = parser.parse_args()
    print(args)

    # Reset the CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    # Create synthetic data
    if args.data_mode == 'entangle':
        data, targets1, targets2, targets3, X, Y, Z1, Zs, Z2 = generate_data(args.num_data, 256, args.dim_info, args.weight_info, args.label_weight_info1, args.label_weight_info2, args.label_weight_info3, seed=args.seed)
    else:
        raise NotImplementedError()
    dataset = MultimodalDataset(data, targets1, targets2, targets3)

    # Dataloader
    num_data = data.shape[1]
    torch.manual_seed(0) # fix the seed for train-test spliting
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(0.8*num_data), num_data-int(0.8*num_data)])
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, batch_size=args.batch_size)
    
    a_s = [0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    seeds = [0, 1, 2]
    # seeds = [0]
    args.gpus = args.device.split(',')
    num_gpus = len(args.device)
    print(f'Testing for a={a_s}')
    print(f'Total process running are {len(a_s) * len(seeds)} spread across GPUs {args.gpus}')
    parameters = [(args, a, seed, train_loader, test_loader, train_dataset, test_dataset) for a, seed in product(a_s, seeds)]
    multiprocessing.set_start_method('spawn', force=True)
    lock = multiprocessing.Lock()

    # Use 'get_context' to create the pool with the 'spawn' method
    with multiprocessing.get_context("spawn").Pool(
        processes=min(len(parameters), multiprocessing.cpu_count()), 
        initializer=init_lock, 
        initargs=(lock,)
    ) as pool:
        results = pool.starmap(wrapper_train_mp, parameters)
        pool.close()
        pool.join()

    print("All results saved.")
