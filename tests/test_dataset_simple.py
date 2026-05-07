from foldnet.data.dataset import get_dataloaders

config = {
    'ss_csv'      : 'data/processed/cb513_ss_labels.csv',
    'splits_json' : 'data/processed/cb513_splits_5fold/splits.json',
    'esm_dir'     : 'data/processed/esm2_embeddings',
    'contact_dir' : None,
    'batch_size'  : 4,
}

train_loader, val_loader = get_dataloaders(fold=0, config=config)
batch = next(iter(train_loader))

print("features   :", batch['features'].shape)
print("ss_labels  :", batch['ss_labels'].shape)
print("contact_map:", batch['contact_map'].shape)
print("mask       :", batch['mask'].shape)
print("Dataset working!")