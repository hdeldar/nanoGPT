"""
Prepare the Quran dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np
import csv

input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    # load the quran dataset
    quran_surehs_path = os.path.join(os.path.dirname(__file__), 'surehs.csv')
    quran_verses_path = os.path.join(os.path.dirname(__file__), 'ayehs.csv')
    with open(quran_surehs_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        surehsdata = list(reader)
    with open(quran_verses_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)
        for verse in data:
            sureh = next((item["order"] for item in surehsdata if item["id"] == verse["sureh_id"]), "")
            verse["sureh_number"] = sureh
        data.sort(key=lambda x: (int(x['sureh_number']), int(x['number'])))

    verses = []

    for i in range(len(data)):
        item = data[i]
        #id = item["id"]
        verse = item["content"]
        #uthmanic = item["uthmanic_text"]
        #ayeh_number = item["number"]
        #sureh_number = item["sureh_number"]
        verses.append(verse)
    all_verses = "\n".join(verses)
    with open(input_file_path, 'w') as f:
        f.write(all_verses)
        

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)




# length of dataset in characters: 701,887
# all the unique characters: 
#  ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىيًٌٍَُِّْٰۖۗۘۙۚۛۜ۞ۡ۩
# vocab size: 57
# train has 631,698 tokens
# val has 70,189 tokens

# run:
# python data/quran/prepare.py
#
# python train.py config/train_quran.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
#
# python sample.py --out_dir=out-quran --device=cpu
