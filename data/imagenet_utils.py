import numpy as np
import pandas as pd
from nltk.corpus import wordnet as wn

def read_hierarchy(label_names, dir='../collected/imagenet/LOC_synset_mapping.txt'):
    map_fname = dir
    synset_ids_1000=np.zeros(1000)
    l=0
    with open(map_fname, 'r') as f:
        for line in f:
            synset_id_s = line.split()[0]
            synset_id=int(synset_id_s[1:])
            synset_ids_1000[l]=synset_id
            l=l+1
    level = 20
    all_list=[["-" for j in range(level)] for i in range(1000)]
    for i in range(1000):
        synset=wn.synset_from_pos_and_offset('n',int(synset_ids_1000[i]))
        hyper_list=[]
        while synset.hypernyms():
            synset = synset.hypernyms()[0]
            hyper_list.append(synset.name())
        hyper_list.append(label_names[i])
        all_list[i][:]=hyper_list[::-1]

    hierarchy = pd.DataFrame(all_list)
    return hierarchy



def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]



def find_all_parents(hierarchy):
    '''
    input:
    hierarchy: pandas dataframe, the hierarchy of labels
    output:
    all_parents: list, all the parent names
    '''
    all_parents = []
    for i in range(len(hierarchy)):
        parent = hierarchy.loc[i].to_list()
        for p in parent[1:]:
            if p and p != "null":
                all_parents.append(p)
    all_parents = unique(all_parents)
    return all_parents