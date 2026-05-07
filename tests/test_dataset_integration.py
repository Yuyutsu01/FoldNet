from foldnet.data.dataset import get_dataloaders

config = {
    'ss_csv'      : 'foldnet/data/processed/cb513_ss_labels.csv',
    'splits_json' : 'foldnet/data/processed/cb513_splits_5fold/splits.json',
    'esm_dir'     : 'foldnet/data/processed/basic_features',
    'contact_dir' : None,
    'batch_size'  : 4,
    'feature_dim' : 25,
}

train_loader, val_loader = get_dataloaders(fold=0, config=config)
batch = next(iter(train_loader))

# 2. Setup Model (using basic feature dim)
from foldnet.models.foldnet import FoldNet
model = FoldNet(
    encoder_type='cnn',
    feature_dim=config['feature_dim'],
    hidden_dim=128,
    num_classes=3
)

print("features   :", batch['features'].shape)
print("ss_labels  :", batch['ss_labels'].shape)
print("contact_map:", batch['contact_map'].shape)
print("mask       :", batch['mask'].shape)
print("Dataset working!")