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


def wrapper_train_mp(args, beta, lam, seed, train_loader, test_loader, train_dataset, test_dataset):
    gpu_id = random.choice(args.gpus)  # Randomly choose from available GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)  # Set CUDA device only for this process
    args.lmd_start = lam
    args.lmd_end = lam 
    args.seed = seed
    args.beta = beta
    args.beta_start = beta
    args.beta_end = beta
    set_seed(seed)
    logs = train_step2(args, train_loader, test_loader, train_dataset, test_dataset)
    log_entry = {f'beta_{beta}_lambda_{lam}_seed_{seed}': logs}
    lock.acquire()
    try:
        log_file_path = f"./results_synthetic_task/{args.data_mode}/step2_training_logs.json"
        if args.custom_name != '':
            log_file_path = f"./results_synthetic_task/{args.data_mode}/step2_training_logs_{args.custom_name}.json"
        with open(log_file_path, 'a') as file:  # Open file in append mode
            json.dump(log_entry, file, indent=4)
            file.write(",\n")  # Add comma and newline for JSON array format
        print(f'Run saved for {log_entry.keys()} in {log_file_path}')
    finally:
        lock.release()

    return log_entry


def train_step2(args, train_loader, test_loader, train_dataset, test_dataset):
    if args.beta == 0:
        step1_name = f'cib_{args.data_mode}_seed0_beta0_kappa{int(args.kappa)}_epoch{args.num_epoch_s1}_dim{args.embed_dim}.tar'
    else:
        step1_name = f'cib_{args.data_mode}_seed0_beta{args.beta}_kappa{int(args.kappa)}_epoch{args.num_epoch_s1}_dim{args.embed_dim}.tar'
    step1_path = os.path.join(f'./results_synthetic_task/{args.data_mode}/models/', step1_name)
    cmib_step1 = MVInfoMaxModel(args.dim_info['X'], args.dim_info['Y'], args.hidden_dim, args.embed_dim, initialization='normal', distribution='vmf', vmfkappa=args.kappa,
                                beta_start_value=args.beta_start, beta_end_value=args.beta_end, beta_n_iterations=8000, beta_start_iteration=0).cuda()
    cmib_step1.load_state_dict(torch.load(step1_path))

    print('Training step 2: Disen model')
    disen = DisenModel(cmib_step1, args.dim_info['X'], args.dim_info['Y'], args.hidden_dim, args.embed_dim, zs_dim=args.embed_dim, initialization = 'normal',
                            lmd_start_value = args.lmd_start, lmd_end_value=args.lmd_end, lmd_n_iterations = 8000, lmd_start_iteration = 0,
                            ortho_norm=args.ortho_norm, condzs=args.condzs, proj=args.proj, usezsx=args.usezsx, apdzs=args.apdzs).cuda()
    disen_optim = optim.Adam(disen.parameters(), lr=args.lr_s2)
    logs = train_Disen(disen, train_loader, disen_optim, train_dataset, test_dataset, num_epoch=args.num_epoch_s2, noise_scale=args.noise_scale, drop_scale=args.drop_scale)
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
    parser.add_argument('--beta_start', type=float, default=0.1)
    parser.add_argument('--beta_end', type=float, default=0.1)
    parser.add_argument('--kappa', type=float, default=1e3)
    parser.add_argument('--lmd_start', type=float, default=0)
    parser.add_argument('--lmd_end', type=float, default=0)
    parser.add_argument('--ortho_norm', type=str2bool, default=True) 
    parser.add_argument('--condzs', type=str2bool, default=True)
    parser.add_argument('--proj', type=str, default='none')
    parser.add_argument('--apdzs', type=str2bool, default=True)
    parser.add_argument('--usezsx', type=str2bool, default=False)

    ## model params
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--embed_dim', type=int, default=100)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epoch_s1', type=int, default=50) # for step 1
    parser.add_argument('--num_epoch_s2', type=int, default=30) # for step 2
    parser.add_argument('--lr_s1', type=float, default=1e-4)
    parser.add_argument('--lr_s2', type=float, default=1e-4)
    parser.add_argument('--custom_name', type=str, default='')
    
    ## self-contrastive aug params
    parser.add_argument('--noise_scale', type=float, default=0.01)
    parser.add_argument('--drop_scale', type=float, default=10)

    # device
    parser.add_argument('--device', type=str, default="0", help="Comma-separated list of available GPUs (e.g., '0,1,2,3')")

    args = parser.parse_args()
    print(args)

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
    
    betas = [args.beta_start]
    lambdas = [0, 0.001, 0.01, 0.1, 1, 10, 100]
    seeds = [0, 1, 2]  
    # seeds = [0]  
    args.gpus = args.device.split(',')
    num_gpus = len(args.device)
    print(f'Testing for beta={betas} and lambdas={lambdas}')
    print(f'Total process running are {len(betas)* len(lambdas) * len(seeds)} spread across GPUs {args.gpus}')
    parameters = [(args, beta, lam, seed, train_loader, test_loader, train_dataset, test_dataset) for beta, lam, seed in product(betas, lambdas, seeds)]
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
    