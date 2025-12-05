import time
import argparse
import torch
import data_setup
from torchvision import transforms


def main(train_dir: str, test_dir: str, batch_size: int, num_workers: int, num_batches: int):
    tfms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    dl_train, _, classes = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=tfms,
        batch_size=batch_size,
        num_workers=num_workers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    iter_dl = iter(dl_train)

    # Warmup
    for _ in range(2):
        try:
            next(iter_dl)
        except StopIteration:
            break

    # Benchmark loop: time to get batch (data load) and time to do a dummy forward
    iter_dl = iter(dl_train)
    for i in range(num_batches):
        t0 = time.time()
        X, y = next(iter_dl)
        load_time = time.time() - t0

        t1 = time.time()
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        # small synthetic model forward (or use your model)
        _ = torch.randn((1, 3, 64, 64), device=device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        comp_time = time.time() - t1

        print(f"batch {i}: load_time={load_time:.3f}s comp_time(~move)={comp_time:.3f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark DataLoader load vs compute time')
    parser.add_argument('--train-dir', type=str, default='data/train')
    parser.add_argument('--test-dir', type=str, default='data/test')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--num-batches', type=int, default=10)
    args = parser.parse_args()

    main(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_batches=args.num_batches
    )