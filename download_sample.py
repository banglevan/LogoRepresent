import pandas as pd
import requests
import os

ROOT = 'data/downloaded'
SPEC = 'logo'
SAVE = os.path.join(ROOT, SPEC)
os.makedirs(SAVE, exist_ok=True)

df = pd.read_excel('products.xlsx')
image_names = []
for i in df.index.values:
    line = df.iloc[i]
    name = line['name']
    url = line['thumbnail_url']
    iname = url.split('/')[-1]
    image_names.append(iname)
    r = requests.get(url, allow_redirects=True)
    open(f'{SAVE}/{iname}', 'wb').write(r.content)
    print(i)

df['image_name'] = image_names
df.to_csv('products.csv', index=False)