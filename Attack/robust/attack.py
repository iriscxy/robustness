import pdb
import matplotlib.pyplot as plt
from datasets import DatasetDict
import json
import torch
import random
import pickle
import nltk
import numpy
from transformers import BartTokenizerFast, BartForConditionalGeneration
import re
from tqdm import tqdm
from nltk.tag.stanford import StanfordNERTagger
import os

st = StanfordNERTagger('stanford-ner-2020-11-17/classifiers/english.all.3class.distsim.crf.ser.gz',
                       'stanford-ner-2020-11-17/stanford-ner.jar')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large')  # 如果用bart-base就用这行
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large").to('cuda')
embeddings = model.model.shared.weight

f = open('../cross_attentions.txt')
ftest = open('test.json')
fw = open('attack.json', 'w')
fw_change_words = open('changewords.txt', 'w')
fantonym = open('antonyms.json')
grads = torch.load('grad.pt')

def get_bpe_substitues(substitutes):
    # substitutes L, k
    substitutes = substitutes[0:4, 0:4]  # maximum BPE candidates

    # find all possible candidates
    all_substitutes = []
    for i in range(substitutes.size(0)):
        if len(all_substitutes) == 0:
            lev_i = substitutes[i]
            all_substitutes = [[int(c)] for c in lev_i]
        else:
            lev_i = []
            for all_sub in all_substitutes:
                for j in substitutes[i]:
                    lev_i.append(all_sub + [int(j)])
            all_substitutes = lev_i

    # all substitutes  list of list of token-id (all candidates)
    c_loss = torch.nn.CrossEntropyLoss(reduction='none')
    all_substitutes = torch.tensor(all_substitutes)  # [ N, L ]
    all_substitutes = all_substitutes[:24].to('cuda')
    N, L = all_substitutes.size()
    word_predictions = model(all_substitutes)[0]  # N L vocab-size

    ppl = c_loss(word_predictions.view(N * L, -1), all_substitutes.view(-1))  # [ N*L ]
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))  # N
    _, word_list = torch.sort(ppl)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []

    for word in word_list:
        text = tokenizer.decode(word)
        final_words.append(text)
    return final_words, word_list

ENGLISH_FILTER_WORDS = [
    'a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
    'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
    'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
    'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
    'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
    "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
    'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
    'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
    'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
    'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
    'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
    "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
    'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
    'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
    'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
    'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
    'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
    'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
    'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
    'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
    'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
    'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
    'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
    "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
    'your', 'yours', 'yourself', 'yourselves', 'have', 'be'
]

cross_attentions = f.readlines()
line = fantonym.readlines()[0]
antonym = json.loads(line)
new_antonym = {}
for key in antonym.keys():
    name = key
    name = name.split(':')[0]
    new_antonym[name] = antonym[key]

cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
means = []
index = -1
lines = ftest.readlines()
my_re = re.compile(r'[A-Za-z]', re.S)
for line in tqdm(lines[:]):
    index += 1
    change_number = 0
    if index % 1000 == 0:
        print(str(index))
        print(str(numpy.mean(means)))
    content = json.loads(line)
    src = content['src']
    tgt = content['tgt']
    token_src = tokenizer(src, max_length=1024, padding=False, truncation=True, return_tensors="pt")
    token_attention_mask = token_src['attention_mask'][0].tolist()
    token_src = token_src['input_ids'].to('cuda')
    logits = model(token_src).logits
    token_src = token_src[0].tolist()

    cross_attention = cross_attentions[index]
    cross_attention = cross_attention.split()
    cross_attention = [float(each) for each in cross_attention]
    sort_index = numpy.argsort(cross_attention)
    sort_index = numpy.flip(sort_index)[:min(100, len(token_src))]
    new_token_src = list(token_src)
    tag_result = st.tag(src.split())
    people_names = [each[0].lower() for each in tag_result if each[1] == 'PERSON' or each[1] == 'LOCATION']

    for temp_index in sort_index:
        if temp_index >= len(token_src) - 1:
            continue
        if token_src[temp_index] <= 4:
            continue
        select_word = tokenizer.decode(token_src[temp_index])
        next_word = tokenizer.decode(token_src[temp_index + 1])
        previous_word = tokenizer.decode(token_src[temp_index - 1])
        res = re.findall(my_re, select_word)
        if ' ' in select_word and len(res) >= 3 and ' ' in next_word:  # single word
            if select_word.strip().lower() in ENGLISH_FILTER_WORDS or select_word.strip().lower() in people_names:
                continue
            ori_embeds = torch.index_select(embeddings, 0,
                                            torch.as_tensor(token_src[temp_index]).to('cuda'))  # [2, 1024])

            probs = logits[0, temp_index].softmax(dim=0)
            values, predictions = probs.topk(10)
            candidates = tokenizer.batch_decode(predictions)
            candidates = [each for each in candidates if
                          select_word.lower() not in each.lower() and each.lower() not in select_word.lower() and each.lower() != select_word.lower()]
            good_candidates = []
            grad = grads[index, temp_index, :]  # (1024,1024)
            for candidate in candidates:
                if ' ' in candidate and candidate.strip().lower() not in ENGLISH_FILTER_WORDS:
                    res = re.findall(my_re, candidate)
                    if len(res) < 3:
                        continue
                    tagged = nltk.pos_tag([candidate.strip().lower()])
                    select_tag = nltk.pos_tag([select_word.strip().lower()])
                    if tagged[0][1] != 'IN' and tagged[0][1] != 'VBD' and tagged[0][1] == select_tag[0][1]:
                        if select_word.strip().lower() in new_antonym.keys():
                            if candidate.strip().lower() not in new_antonym[select_word.strip().lower()].split('|'):
                                good_candidates.append(candidate)
                        else:
                            good_candidates.append(candidate)
            if len(good_candidates) == 0:
                continue
            elif len(good_candidates) == 1:
                candidate_ids = tokenizer.encode(good_candidates[0], add_special_tokens=False)[0]
                fw_change_words.write(select_word.strip() + good_candidates[0] + '\n')
                new_token_src[temp_index] = candidate_ids
                change_number += 1
            elif len(good_candidates) >= 1:
                candidate_ids = tokenizer.encode(''.join(good_candidates), add_special_tokens=False,
                                                 return_tensors="pt").to('cuda').squeeze(0)
                assert candidate_ids.shape[0] == len(good_candidates)
                embeds = torch.index_select(embeddings, 0, candidate_ids)  # [2, 1024])
                grad = torch.unsqueeze(grad, 0)

                similarities = cos(embeds - ori_embeds, -grad)
                index_sorted = torch.argsort(similarities)[0]
                new_token_src[temp_index] = candidate_ids[index_sorted]
                fw_change_words.write(select_word.strip() + good_candidates[index_sorted] + '\n')
                change_number += 1
        elif ' ' in select_word and len(res) >= 3 and ' ' not in next_word and False not in [each not in next_word for
                                                                                             each in
                                                                                              "!\"#$%&'()*+, -./:;<=>?@[\]^_`{|}~"]:
            if len(select_word) < 3:
                continue
            if select_word.strip().lower() in people_names:
                continue
            start_index = temp_index
            end_index = temp_index + 1
            while ' ' not in next_word and False not in [each not in next_word for each in
                                                          "!\"#$%&'()*+, -./:;<=>?@[\]^_`{|}~"]:
                end_index = end_index + 1
                if end_index >= len(token_src) - 1:
                    break
                next_word = tokenizer.decode(token_src[end_index])
            if False not in [each not in next_word for each in "!\"#$%&'()*+, -./:;<=>?@[\]^_`{|}~"]:
                continue
            original_word = tokenizer.decode(token_src[start_index:end_index])
            res = re.findall(my_re, original_word)
            if len(res) < 3 or end_index - start_index >= 5:  # single word
                continue

            substitutes = []
            for search_index in range(start_index, end_index):
                probs = logits[0, search_index].softmax(dim=0)
                values, predictions = probs.topk(10)
                substitutes.append(predictions)
            substitutes = torch.stack(substitutes, 0)

            candidates, word_list = get_bpe_substitues(substitutes)
            good_word_list = []
            good_candidates = []
            for can_index, candidate in enumerate(candidates):
                if ' ' in candidate and candidate.strip().lower() not in ENGLISH_FILTER_WORDS:
                    res = re.findall(my_re, candidate)
                    if len(res) < 3:
                        continue
                    tagged = nltk.pos_tag([candidate.strip().lower()])
                    ori_tagged = nltk.pos_tag([original_word.strip().lower()])
                    if tagged[0][1] != 'IN' and tagged[0][1] != 'VBD' and candidate.lower() != original_word.lower() and \
                            ori_tagged[0][1] == tagged[0][1]:
                        if select_word in new_antonym.keys():
                            if candidate.strip().lower() not in new_antonym[select_word].split('|'):
                                good_candidates.append(candidate)
                                good_word_list.append(word_list[can_index])
                        else:
                            good_candidates.append(candidate)
                            good_word_list.append(word_list[can_index])
            if len(good_candidates) == 0:
                continue
            remove_index = random.randint(0, min(5, len(good_candidates) - 1))
            two_token = good_word_list[remove_index]
            two_token = [each.item() for each in two_token]
            new_token_src[start_index:end_index] = two_token
            fw_change_words.write(original_word.strip() + good_candidates[remove_index] + '\n')
            change_number += 1
        elif ' ' not in select_word and len(res) >= 3:
            if len(select_word) < 3:
                continue
            if select_word.strip().lower() in people_names:
                continue
            start_index = temp_index - 1
            end_index = temp_index + 1
            while ' ' not in previous_word and '.' not in previous_word and False not in [each not in next_word for
                                                                                          each in
                                                                                          "!\"#$%&'()*+, -./:;<=>?@[\]^_`{|}~"]:
                start_index = start_index - 1
                if start_index == 0:
                    break
                previous_word = tokenizer.decode(token_src[start_index])
            if True in [each in previous_word for each in "!\"#$%&'()*+, -./:;<=>?@[\]^_`{|}~"]:
                continue
            while ' ' not in next_word and '.' not in next_word and False not in [each not in next_word for each in
                                                                                    "!\"#$%&'()*+, -./:;<=>?@[\]^_`{|}~"]:
                end_index = end_index + 1
                if end_index >= len(token_src) - 1:
                    break
                next_word = tokenizer.decode(token_src[end_index])
            if True in [each in next_word for each in "!\"#$%&'()*+, -./:;<=>?@[\]^_`{|}~"]:
                continue
            original_word = tokenizer.decode(token_src[start_index:end_index])
            res = re.findall(my_re, original_word)
            if len(res) < 3:
                continue

            substitutes = []
            for search_index in range(start_index, end_index):
                probs = logits[0, search_index].softmax(dim=0)
                values, predictions = probs.topk(10)
                substitutes.append(predictions)
            substitutes = torch.stack(substitutes, 0)

            candidates, word_list = get_bpe_substitues(substitutes)
            good_word_list = []
            good_candidates = []
            for can_index, candidate in enumerate(candidates):
                if ' ' in candidate and candidate.strip().lower() not in ENGLISH_FILTER_WORDS:
                    res = re.findall(my_re, candidate)
                    if len(res) < 3:
                        continue
                    tagged = nltk.pos_tag([candidate.strip().lower()])
                    ori_tagged = nltk.pos_tag([original_word.strip().lower()])
                    if tagged[0][1] != 'IN' and tagged[0][1] != 'VBD' and candidate.lower() != original_word.lower() and \
                            ori_tagged[0][1] == tagged[0][1]:
                        if select_word in new_antonym.keys():
                            if candidate.strip().lower() not in new_antonym[select_word].split('|'):
                                good_candidates.append(candidate)
                                good_word_list.append(word_list[can_index])
                        else:
                            good_candidates.append(candidate)
                            good_word_list.append(word_list[can_index])
            if len(good_candidates) == 0:
                continue
            remove_index = random.randint(0, min(5, len(good_candidates) - 1))
            two_token = good_word_list[remove_index]
            two_token = [each.item() for each in two_token]
            new_token_src[start_index:end_index] = two_token
            fw_change_words.write(original_word.strip() + good_candidates[remove_index] + '\n')
            change_number += 1
    assert len(new_token_src) == len(token_src)
    means.append(change_number)
    new_decode_str = ''.join(tokenizer.batch_decode(new_token_src[1:-1]))

    decode_str = ''.join(tokenizer.batch_decode(token_src))
    content['input_ids'] = token_src
    content['attention_mask'] = token_attention_mask
    content['aug_input_ids'] = new_token_src
    content['aug_src'] = new_decode_str
    assert len(token_src) == len(new_token_src) and len(token_attention_mask) == len(token_src)
    json.dump(content, fw)
    fw.write('\n')
