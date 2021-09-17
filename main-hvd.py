#! /homes/zhengchun.liu/usr/miniconda3/envs/hvd/bin/python

from model import CookieAE128ch, CookieAE16ch
import torch, argparse, os, time, sys, shutil, logging
from torch.utils.data import DataLoader
from data import CookieAEDataSet, get_validation_ds
import numpy as np
import horovod.torch as hvd

parser = argparse.ArgumentParser(description='Cokkie AE')
parser.add_argument('-gpus',   type=str, default="", help='list of GPUs to use')
parser.add_argument('-expName',type=str, default="debug", help='Experiment name')
parser.add_argument('-lr',     type=float,default=3e-4, help='learning rate')
parser.add_argument('-mbsz',   type=int, default=128, help='mini batch size')
parser.add_argument('-maxep',  type=int, default=400, help='max training epoches')
parser.add_argument('-imgsz',  type=int, default=512, help='image size')
parser.add_argument('-ch',     type=int, default=128, help='channel size')

args, unparsed = parser.parse_known_args()

if len(unparsed) > 0:
    print('Unrecognized argument(s): \n%s \nProgram exiting ... ... ' % '\n'.join(unparsed))
    exit(0)

if len(args.gpus) > 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

def main(args, itr_out_dir):
    torch_devs = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = CookieAEDataSet(ch=args.ch)
    
    # Partition dataset among workers using DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())

    data_loader = DataLoader(train_dataset, batch_size=args.mbsz//hvd.size(), pin_memory=True, drop_last=True, sampler=train_sampler)

    X_mb_val, y_mb_val = get_validation_ds(ch=args.ch, dev=torch_devs)

    if args.ch == 128:
        model = CookieAE128ch()
    elif args.ch == 16:
        model = CookieAE16ch()
    else:
        print('unrecognizable ch: %d' % args.ch)

    model = model.cuda()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * (args.mbsz//16), weight_decay=0) 
    # Add Horovod Distributed Optimizer
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    # Broadcast parameters from rank 0 to all other processes.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    time_on_training = 0
    for epoch in range(args.maxep):
        ep_tick = time.time()
        time_comp, time_opt = 0, 0
        for _iter, (X_mb, y_mb) in enumerate(data_loader):
            it_comp_tick = time.time()
            optimizer.zero_grad()
            pred = model.forward(X_mb.to(torch_devs))
            loss = criterion(pred, y_mb.to(torch_devs))
            loss.backward()
            opt_tick = time.time()
            optimizer.step() 

            time_opt  += 1000 * (time.time() - opt_tick)
            time_comp += 1000 * (time.time() - it_comp_tick)

        time_e2e = 1000 * (time.time() - ep_tick)
        time_on_training += time_e2e
        if hvd.rank() !=0: continue

        _prints = 'rank %2d [Train] @ %.1f Epoch: %05d, loss: %.4f, elapse: %.2fms/epoch (computation=%.1fms, %.2f%%, sync-and-opt: %.1fms)' % (\
                   hvd.local_rank(), time.time(), epoch, loss.cpu().detach().numpy(), time_e2e, time_comp, 100*time_comp/time_e2e, time_opt)
        logging.info(_prints)

        with torch.no_grad():
            pred_val = model.forward(X_mb_val)
            loss_val = torch.nn.MSELoss()(pred_val, y_mb_val)
        _prints = '[Validation] @ %.1f Epoch: %05d, loss: %.4f' % (\
                   time.time(), epoch, loss_val.cpu().detach().numpy(), )
        logging.info(_prints)
        torch.save(model.state_dict(), "%s/hvd-ep%03d.pth" % (itr_out_dir, epoch))
    logging.info("rank %2d Trained for %3d epoches, each with %d steps (BS=%d) took %.3f seconds" % (hvd.rank(), \
                 args.maxep, len(data_loader), X_mb.shape[0], time_on_training*1e-3))

if __name__ == "__main__":
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())
    if args.mbsz % hvd.size() != 0:
        print("global batch size is not divisible by the number of workers")
        exit(0)

    itr_out_dir = args.expName + '-itrOut'
    if hvd.rank() == 0:
        if os.path.isdir(itr_out_dir): 
            shutil.rmtree(itr_out_dir)
        os.mkdir(itr_out_dir) # to save temp output

    hvd.allreduce(torch.rand(1)) # behaves like barrier dealing race condition
    logging.basicConfig(filename="%s/CookieNetAE-HVD.log" % (itr_out_dir, ), level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    main(args, itr_out_dir)
