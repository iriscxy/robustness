import pdb
import random
from transformers import pipeline
from datasets import load_dataset
import json

dataset = load_dataset("ccdv/cnn_dailymail", '3.0.0')
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

train_dataset = dataset['train']
fw=open('train.json','w')
for case in train_dataset:
    ARTICLE = case['article']
    highlights = case['highlights']
    content={}
    content['src']=ARTICLE
    content['tgt']=highlights
    json.dump(content,fw)
    fw.write('\n')

train_dataset = dataset['validation']
fw=open('validation.json','w')
for case in train_dataset:
    ARTICLE = case['article']
    highlights = case['highlights']
    content={}
    content['src']=ARTICLE
    content['tgt']=highlights
    json.dump(content,fw)
    fw.write('\n')

train_dataset = dataset['test']
fw=open('test.json','w')
for case in train_dataset:
    ARTICLE = case['article']
    highlights = case['highlights']
    content={}
    content['src']=ARTICLE
    content['tgt']=highlights
    json.dump(content,fw)
    fw.write('\n')