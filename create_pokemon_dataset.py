from tqdm import tqdm
import os
import pandas as pd
import csv

entries = os.listdir('Bulbasaur/')
df1 = pd.DataFrame(entries)
df1.columns = ['File-ID']
df1['File-ID'] = 'Bulbasaur/' + df1['File-ID'].astype(str)
df1['label'] = '0'

entries = os.listdir('Charmander/')
df2 = pd.DataFrame(entries)
df2.columns = ['File-ID']
df2['File-ID'] = 'Charmander/' + df2['File-ID'].astype(str)
df2['label'] = '1'
df1 = df1.append(df2,ignore_index=True)

entries = os.listdir('Eevee/')
df3 = pd.DataFrame(entries)
df3.columns = ['File-ID']
df3['File-ID'] = 'Eevee/' + df3['File-ID'].astype(str)
df3['label'] = '2'
df1 = df1.append(df3,ignore_index=True)

entries = os.listdir('Pikachu/')
df4 = pd.DataFrame(entries)
df4.columns = ['File-ID']
df4['File-ID'] = 'Pikachu/' + df4['File-ID'].astype(str)
df4['label'] = '3'
df1 = df1.append(df4,ignore_index=True)

entries = os.listdir('Squirtle/')
df5 = pd.DataFrame(entries)
df5.columns = ['File-ID']
df5['File-ID'] = 'Squirtle/' + df5['File-ID'].astype(str)
df5['label'] = '4'
df1 = df1.append(df5,ignore_index=True)

print(df1)
export_csv = df1.to_csv(r'pokemon-dataset.csv', index = None, header=False)
