import torch
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils import DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import BatchSampler, DataLoader, IterableDataset, RandomSampler


# Custom Dataset
class SequenceDataset(Dataset):
    def __init__(self, start, end):
        self.data = torch.arange(start, end + 1)  # Sequence from start to end
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# Initialize the dataset and DataLoader
dataset = SequenceDataset(1, 50)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

# Iterate through the DataLoader
set_seed(42)
# print(isinstance(dataloader.sampler, RandomSampler))
# print(type(dataloader.batch_sampler))
# u = getattr(dataloader.batch_sampler, "sampler", None)
# print(u)
dataloader_config = DataLoaderConfiguration()
dataloader_config.split_batches=True
dataloader_config.use_seedable_sampler  = True
dataloader_config.dispatch_batches = True
kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[kwargs], step_scheduler_with_optimizer=False, dataloader_config=dataloader_config)

dataloader = accelerator.prepare(dataloader)
print(dataloader.get_sampler())

for batch in dataloader:
    print(batch)