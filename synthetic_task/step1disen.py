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


def wrapper_train_mp(args, beta, seed, train_loader, test_loader, train_dataset, test_dataset):
    gpu_id = random.choice(args.gpus)  # Randomly choose from available GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)  # Set CUDA device only for this process
    args.beta_start = beta
    args.beta_end = beta 
    args.seed = seed
    set_seed(seed)
    logs = train_mp(args, train_loader, test_loader, train_dataset, test_dataset)
    log_entry = {f'beta_{beta}_seed_{seed}': logs}
    lock.acquire()
    try:
        log_file_path = f"./results_synthetic_task/{args.data_mode}/step1_training_logs.json"
        if not args.debug_mode:
            with open(log_file_path, 'a') as file:  # Open file in append mode
                json.dump(log_entry, file, indent=4)
                file.write(",\n")  # Add comma and newline for JSON array format
    finally:
        lock.release()

    return log_entry


def train_mp(args, train_loader, test_loader, train_dataset, test_dataset):
    if args.simclr:
        args.beta_start = 0
        args.beta_end = 0

    if args.beta_start == args.beta_end:
        beta_name = f'beta{args.beta_start}'
    else:
        beta_name = f'beta{args.beta_start}_{args.beta_end}'
    step1_name = f'cib_{args.data_mode}_seed{args.seed}_{beta_name}_kappa{int(args.kappa)}_epoch{args.num_epoch_s1}_dim{args.embed_dim}.tar'
    print('Saving model at', step1_name)
    out_dir = f'./results_synthetic_task/{args.data_mode}/models/'
    os.makedirs(out_dir, exist_ok=True)
    step1_path = os.path.join(f'./results_synthetic_task/{args.data_mode}/models/', step1_name)

    print(f'Training step 1: CIB model on GPU {os.environ["CUDA_VISIBLE_DEVICES"]} for beta:', args.beta_start, 'seed:', args.seed)
    cmib = MVInfoMaxModel(args.dim_info['X'], args.dim_info['Y'], args.hidden_dim, args.embed_dim, initialization = 'normal', distribution='vmf', vmfkappa=args.kappa,
                        beta_start_value = args.beta_start, beta_end_value=args.beta_end, beta_n_iterations = 8000, beta_start_iteration = 0, head=args.head, simclr= args.simclr).cuda()
    cmib_optim = optim.Adam(cmib.parameters(), lr=args.lr_s1)
    logs = train(cmib, train_loader, cmib_optim, train_dataset, test_dataset, num_epoch=args.num_epoch_s1)
    if not args.debug_mode:
        torch.save(cmib.state_dict(), step1_path)
        print(f'Save model to {step1_path}')

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
    parser.add_argument('--beta_start', type=float)
    parser.add_argument('--beta_end', type=float)
    parser.add_argument('--kappa', type=float, default=1e3)
    parser.add_argument('--ortho_norm', type=str2bool, default=True) 
    parser.add_argument('--condzs', type=str2bool, default=True)
    parser.add_argument('--proj', type=str2bool, default=False)
    parser.add_argument('--apdzs', type=str2bool, default=True)
    parser.add_argument('--usezsx', type=str2bool, default=False)
    parser.add_argument('--simclr', type=str2bool, default=False)
    parser.add_argument('--head', type=str, default='none')

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
    
    ## self-contrastive aug params
    parser.add_argument('--noise_scale', type=float, default=0.01)
    parser.add_argument('--drop_scale', type=float, default=10)
    parser.add_argument('--debug_mode', type=str2bool, default=False)

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
    
    betas = [0, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 300.0, 500.0, 1000.0]
    seeds = [0, 1, 2]  # Example seed values
    # seeds = [0] 

    args.gpus = args.device.split(',')
    num_gpus = len(args.device)
    parameters = [(args, beta, seed, train_loader, test_loader, train_dataset, test_dataset) for beta, seed in product(betas, seeds)]
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
