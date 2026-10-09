import os, zipfile, glob
root = glob.glob('/kaggle/input/*/competition_data/images') or glob.glob('/kaggle/input/**/images', recursive=True)
print('roots', root)
src = root[0]
for split in ['test', 'train']:
    d = os.path.join(src, split)
    files = sorted(os.listdir(d))
    with zipfile.ZipFile(f'/kaggle/working/{split}.zip', 'w', zipfile.ZIP_STORED) as z:
        for f in files:
            z.write(os.path.join(d, f), f'{split}/{f}')
    print(split, len(files), os.path.getsize(f'/kaggle/working/{split}.zip') / 1e6, 'MB')
