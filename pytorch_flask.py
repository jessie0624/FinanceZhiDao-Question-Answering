import flask
from flask import Flask, render_template, session, request, redirect, url_for,jsonify
# from app import app
import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from transformers import AdamW, WarmupLinearSchedule
from transformers import WEIGHTS_NAME, BertConfig, BertForSequenceClassification, BertTokenizer
from transformers.data.processors.utils import DataProcessor, InputExample,InputFeatures

import pandas as pd
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.linalg import norm
logger = logging.getLogger(__name__)
 


def convert_one_example_to_features(examples, tokenizer, max_length=512, pad_token=0, pad_token_segment_id=0, mask_padding_with_zero=True):
    
    features = []
    for (ex_index, example) in enumerate(examples):
        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
            truncate_first_sequence=False  # We're truncating the first sequence in priority
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)
        features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              label=None))
    return features

class FaqProcessor(DataProcessor):
    
    def _create_one_example(self, title, replies):
        examples = []
        num_examples = len(replies)
        for i in range(num_examples):
            examples.append(InputExample(guid=i, text_a=str(title), text_b=str(replies[i]), label=None))
        return examples
    
    def prepare_replies(self, path):
        df = pd.read_csv(path)
        df = df.fillna(0)
        replies = [str(t) for t in df['reply'].tolist()]
        return replies

def getSimiTitleAnswers(new_title, all_title, total_df):

    def tfidf_similarity(s1, s2):
        def add_space(s):
            return ' '.join(list(s))

        # 将字中间加入空格
        s1, s2 = add_space(s1), add_space(s2)
        # 转化为TF矩阵
        cv = TfidfVectorizer(tokenizer=lambda s: s.split())
        corpus = [s1, s2]
        vectors = cv.fit_transform(corpus).toarray()
        # 计算TF系数
        return np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))

    sim_reply = []
    for tt in all_title:
        if tfidf_similarity(tt, new_title) > 0.5:
            print('sim title:', tt + '\n')
            tt_replies = total_df[total_df.best_title==tt].reply.tolist()
            
            if len(tt_replies) > 1:
                for i in tt_replies:
                    if i not in sim_reply:
                        sim_reply.append(i)
            else:
                if tt_replies not in sim_reply:
                    sim_reply.extend(tt_replies)
    return sim_reply

def predict(title, replies, tokenizer, model, device):
    processor = FaqProcessor()
    examples = processor._create_one_example(title, replies)
    features = convert_one_example_to_features(examples, tokenizer)
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    with torch.no_grad():
        inputs = {'input_ids':      all_input_ids.to(device),
                    'attention_mask': all_attention_mask.to(device),
                    'token_type_ids': all_token_type_ids.to(device)}
        outputs = model(**inputs)
        logits = outputs[0]
        score = F.softmax(logits,dim=1)
        print(score)
        values,index = score.sort(0,descending=True)

    recommend_answers = []
    for i in range(values.shape[0]):
        if values[i][1] > 0.9 and len(recommend_answers) < 5:
            recommend_answers.append(replies[index[i][1]])
            
    print('\nQuestion: ', title + '\n')
    print('\nAll answers for predicts: \n')
    for rep in replies:
        print('\t'+ rep + '\n')
    if len(recommend_answers) == 0:
        print('No recommend answers!')
    else:
        print('Recommend_answers: \n')
        for i in recommend_answers:
            print('\t' + i + '\n')
    #ret_str = ' <br />'.join(recommend_answers)
    return recommend_answers
       

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained('checkpoint-18000')
model.eval()
model.to(device)
tokenizer = BertTokenizer.from_pretrained('checkpoint-18000')

train_df = pd.read_csv('dataPreClean/method-3/train.csv', encoding='utf-8-sig',sep='\t')[['best_title','is_best','reply']]
test_df = pd.read_csv('dataPreClean/method-3/test.csv', encoding='utf-8-sig',sep='\t')[['best_title','is_best','reply']]
pred_df = pd.read_csv('dataPreClean/method-3/test.csv', encoding='utf-8-sig',sep='\t')[['best_title','is_best','reply']]
total_df = pd.concat([train_df[train_df.is_best == 1], test_df[test_df.is_best==1]], axis=0)[['best_title','is_best','reply']]
print(total_df.iloc[-2:-1,:])
all_title = total_df.best_title.drop_duplicates().tolist()

app = Flask(__name__)
    
app.secret_key = 'F12Zr47j\3yX R~X@H!jLwf/T'
    
@app.route("/")
def hello_world():
    return render_template('FQA.html')

@app.route('/fqa', methods=['GET'])
def fqa():
    while request.args.get('title'):
        title = str(request.args.get('title'))
        if request.args.get('replies'): ## 如果有备选答案 则进行切分
            replies = request.args.get('replies').split('\r\n') 
        else:## 如果没有备选答案,则根据问题文本相似度查找与title相似的问题,找到这些问题对应的正确答案.
            replies = getSimiTitleAnswers(title, all_title, total_df)
        print(title)
        print(replies)
        if len(replies) == 0:
            return render_template('FQA.html', message='No answers recommend!')
        ret = predict(title, replies, tokenizer, model, device)
        print(ret)
        return render_template('FQA.html', message=ret, title=title, replies=replies)
    else:
        return render_template('FQA.html')


if __name__ == '__main__':

    app.run(port=5000, debug=True)



