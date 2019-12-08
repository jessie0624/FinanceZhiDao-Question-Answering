# FinanceZhiDao_FQA

This is an QABot base on 'financezhidao_filter.csv' from https://github.com/SophonPlus/ChineseNlpCorpus 

The model are base on hunggingface's transformers https://github.com/huggingface/transformers

For chinese version introduction
https://blog.csdn.net/m0_37531129/article/details/102616868

If you want to repeat the repo, pls following steps:

   1. Please install transformers first refering https://huggingface.co/transformers/installation.html
   2. Please download finacezhidao.csv from https://github.com/SophonPlus/ChineseNlpCorpus  and loaded to preCleanDatafolder then run data clean for finace_zhidao_filter.ipynb to get cleaned training and test data
   3. Please run bert_update.py using following cmd to get trained model.
      You can also skip this step, I provid a trained model for you to use run evaluation and test, please download the model from here  https://pan.baidu.com/s/1QE1eNr9kd9hDbQK4KKbzhw password: 3uja
   4. Run bert_update.py --do_predict, you will get the terminal version interaction (it is clumsiness,but this is basic version for test)
   5. Run python_flask.py to get html version interaction, this is friendly for user (*￣︶￣ *).

Let me know if you have any question or any suggestions!

