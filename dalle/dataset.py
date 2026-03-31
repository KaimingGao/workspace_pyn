
import torch
import PIL
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from torch.utils.data.distributed import DistributedSampler
from transformers import default_data_collator
from d_vae.utils import map_pixels


#
# resize image + center crop + laplas shift.
#
def preprocess(image, image_size=256):
    s = min(image.size)
    
    if s < image_size:
        raise ValueError(f'min dim for image {s} < {image_size}')
        
    r = image_size / s

    s = (round(r * image.size[1]), round(r * image.size[0]))

    image = TF.resize(image, s, interpolation=PIL.Image.LANCZOS)

    image = TF.center_crop(image, output_size=2*[image_size])
    #
    # [3, 256, 256] -> [1, 3, 256, 256]
    #
    image = torch.unsqueeze(T.ToTensor()(image), 0)
    #
    # [1, 1, 256, 256] -> [1, 3, 256, 256]
    #
    if image.shape[1] == 1:
        image = image.repeat(1, 3, 1, 1)
    #
    # shift image.
    #
    return map_pixels(image).squeeze()



#
# return 
#
class DVAEDataset(torch.utils.data.Dataset):
    """
    line: [image id] [image file]
    """
    def __init__(self, data_file, image_size=256):
        super(DVAEDataset, self).__init__()

        self.lines = []
        with open(data_file, 'r', encoding='utf-8') as reader:
            for line in reader:
                arr = line.strip().split("\t")
                if len(arr) != 2:
                    continue
                self.lines.append(arr[1])


    def __getitem__(self, i):
        image = PIL.Image.open(self.lines[i])
        image = preprocess(image)
        return {"image": image}


    #
    # return size of dataset.
    #
    def __len__(self):
        return len(self.lines)





def get_dataloader(data_file=None, image_size=256, batch_size=1024, world_size=-1, global_rank=-1, worker_num=8, prefetch_factor=2, drop_last=True, shuffle=True, seed=42):
    #
    # create dvae dataset.
    #
    dataset = DVAEDataset(data_file=data_file, image_size=image_size)

    if global_rank <= 0:
        print("dataset size:", len(dataset), flush=True)

    #
    # support DP and DDP
    #
    if world_size < 0 or global_rank < 0:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size, 
            num_workers=worker_num,
            prefetch_factor=prefetch_factor,
            collate_fn=default_data_collator,
            shuffle=shuffle,
            pin_memory=True,
            drop_last=drop_last,
        )
        return data_loader
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=DistributedSampler(dataset, num_replicas=world_size, rank=global_rank, shuffle=shuffle, seed=seed),
            batch_size=batch_size, 
            num_workers=worker_num,
            prefetch_factor=prefetch_factor,
            collate_fn=default_data_collator,
            pin_memory=True,
            drop_last=drop_last,
        )
        return data_loader



if __name__ == '__main__':
    data_file = '/share/ad/gaokaiming/workspace_pyn/pytorch/data/rel.txt'

    data_loader = get_dataloader(data_file=data_file, batch_size=2)

    for batch in data_loader:
        print(batch)
        break

