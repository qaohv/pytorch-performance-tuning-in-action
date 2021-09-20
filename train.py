import argparse
import os
import time

import albumentations as A
import numpy as np
import torch

from distutils.util import strtobool

from apex import amp
from apex.optimizers import FusedAdam
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from dataset import PersonSegmentationDataset
from model import UnetResnet34

amp.register_float_function(torch, 'sigmoid')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=None,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")

    parser.add_argument('--df-path', required=True, type=str, help='Path to csv with meta-information')
    parser.add_argument('--root-path', required=True, type=str, help='Path to dataset root')

    parser.add_argument('--pretrained-model', default=True, type=strtobool,
                        help='Use imagenet pretrained backbone for Unet')
    parser.add_argument('--enable-bias-decoder', default=True, type=strtobool,
                        help='Enable bias for convolutions in decoder')

    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train (default: 30)')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training (default: 8)')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of workers for dataloader')
    parser.add_argument('--pin-memory', type=strtobool, default=False, help='Enable pin_memory option for dataloader')

    parser.add_argument('--detect-anomaly', type=strtobool, default=False,
                        help='Sets mode for torch.autograd.set_detect_anomaly')
    parser.add_argument('--show-debug-api-status', type=strtobool, default=True,
                        help='Prints debug api status')

    parser.add_argument('--speedup-zero-grad', type=strtobool, default=False,
                        help='Use parameter.grad = None instead of optimizer.zero_grad() or net.zero_grad()')

    parser.add_argument('--enable-gradient-checkpointing', type=strtobool, default=False,
                        help='Enable gradient checkpointing technique for model.')

    parser.add_argument('--enable-cudnn-benchmark', type=strtobool, default=False,
                        help='Enable cudnn benchmark to choose best conv algorithm')

    parser.add_argument('--mixed-precision-mode', type=str, default="O0", choices=["O0", "O1", "O2"],
                        help='Disable or enable mixed precision training.')
    parser.add_argument('--use-fused-adam', type=strtobool, default=False,
                        help='Use apex.optimizers.FusedAdam instead of torch.optim.Adam')

    args = parser.parse_args()

    if args.enable_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(mode=bool(args.detect_anomaly))

    if args.show_debug_api_status:
        print(f"Anomaly detection enabled? : {torch.is_anomaly_enabled()}")
        print(f"Profiler enabled? : {torch.autograd._profiler_enabled()}")

    dist_training = int(os.environ.get("WORLD_SIZE", 1)) > 1

    if dist_training:
        # for dist training is important to make all operations deterministic
        torch.manual_seed(42)
        np.random.seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        torch.distributed.init_process_group(backend="nccl")

    device = torch.device(f"cuda:{args.local_rank or 0}")
    torch.cuda.set_device(device)

    train_dataset = PersonSegmentationDataset(
        df_path=args.df_path,
        root_path=args.root_path,
        transforms=A.Compose([
            A.RandomRotate90(p=.5),
            A.Blur(p=.5),
            A.ElasticTransform(p=.5),
        ])
    )
    if dist_training:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            sampler=DistributedSampler(dataset=train_dataset)
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )

    net = UnetResnet34(args.pretrained_model, args.enable_bias_decoder, args.enable_gradient_checkpointing).to(device)
    optimizer = FusedAdam(net.parameters(), lr=1e-3) if args.use_fused_adam else Adam(net.parameters(), lr=1e-3)
    net, optimizer = amp.initialize(net, optimizer, opt_level=args.mixed_precision_mode)

    if dist_training:
        net = torch.nn.parallel.DistributedDataParallel(net,
                                                        device_ids=[args.local_rank],
                                                        output_device=args.local_rank)
    criterion = BCELoss()

    epoch_timings = []
    for epoch in range(args.epochs):
        train_loss = []
        epoch_start = time.time()

        for images, masks in tqdm(train_loader):
            if args.speedup_zero_grad:
                for param in net.parameters():
                    param.grad = None
            else:
                optimizer.zero_grad()

            images, masks = images.to(device), masks.to(device)
            output = net(images)
            loss = criterion(output.squeeze(1), masks.squeeze(1))

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            train_loss.append(loss.item())
            optimizer.step()

        epoch_finish = time.time()
        epoch_timings.append(epoch_finish - epoch_start)
        print(f"Epoch: {epoch}, train loss: {np.mean(train_loss)}, time: {epoch_finish - epoch_start}")

    print(f"Total epochs number: {args.epochs}, "
          f"mean epoch time: {np.mean(epoch_timings)}, "
          f"std epoch time: {np.std(epoch_timings)}")
