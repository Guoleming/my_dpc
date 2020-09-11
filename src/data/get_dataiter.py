from src.data.dataset_3d import Kinetics400_full_3d, UCF101_3d, Phoenix14_3d
import torch.utils.data as data
from torchvision import transforms
from src.data.augmentation import *


def get_data(args, mode='train'):
    """Set transforms, and get data iterator.
    """
    print('Loading "%s" data for "%s" ...' %(args.dataset, mode))
    if args.dataset == 'ucf101':   # designed for ucf101, short size=256, rand crop to 224x224 then scale to 128x128
        transform = transforms.Compose([
            RandomHorizontalFlip(consistent=True),
            RandomCrop(size=224, consistent=True),
            Scale(size=(args.img_dim, args.img_dim)),
            RandomGray(consistent=False, p=0.5),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0),
            ToTensor(),
            Normalize()
        ])
    elif args.dataset == 'k400':  # designed for kinetics400, short size=150, rand crop to 128x128
        transform = transforms.Compose([
            RandomSizedCrop(size=args.img_dim, consistent=True, p=1.0),
            RandomHorizontalFlip(consistent=True),
            RandomGray(consistent=False, p=0.5),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0),
            ToTensor(),
            Normalize()
        ])
    elif args.dataset == "phoenix14":
        transform = transforms.Compose([
            RandomSizedCrop(size=args.img_dim, consistent=True, p=1.0),
            ToTensor(),
            Normalize()
        ])

    if args.dataset == 'k400':
        use_big_K400 = args.img_dim > 140
        dataset = Kinetics400_full_3d(mode=mode,
                                      transform=transform,
                                      seq_len=args.seq_len,
                                      num_seq=args.num_seq,
                                      downsample=5,
                                      big=use_big_K400)
    elif args.dataset == 'ucf101':
        dataset = UCF101_3d(mode=mode,
                            transform=transform,
                            seq_len=args.seq_len,
                            num_seq=args.num_seq,
                            downsample=args.ds)

    elif args.dataset == "phoenix14":
        dataset = Phoenix14_3d(mode=mode,
                               transform=transform,
                               seq_len=args.seq_len,
                               num_seq=args.num_seq,
                               downsample=args.ds)
    else:
        raise ValueError('dataset not supported')

    if mode == 'train':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=32,
                                      pin_memory=True,
                                      drop_last=True,
                                      collate_fn=dataset.collate_fn)
    elif mode == 'val':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=32,
                                      pin_memory=True,
                                      drop_last=True,
                                      collate_fn=dataset.collate_fn)
    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader

if __name__ == "__main__":
    from config.options import args

    phoenix_data = get_data(args)
    for i, data in enumerate(phoenix_data):
        if i > 5:
            break
        print(data.shape)


