# FinanceZhiDao_FQA

This is an QABot base on 'financezhidao_filter.csv' from https://github.com/SophonPlus/ChineseNlpCorpus 

The model are base on hunggingface's transformers https://github.com/huggingface/transformers

For chinese version introduction
https://blog.csdn.net/m0_37531129/article/details/102616868

If you want to repeat the repo, pls following steps:

   1. Please install transformers first refering https://huggingface.co/transformers/installation.html
   2. (optional) If you want to try training the model by yourself 
      - please run preCleanData.sh to get cleaned training and test data
      - please run train.sh for training.
      You can also skip this step, I provid a trained model for you to use run evaluation and test, please download the model from here  https://pan.baidu.com/s/1QE1eNr9kd9hDbQK4KKbzhw password: 3uja
   3. Run testDemo.sh for testing, you will get the result for default demo in code.
   4. Run testUsermode.sh for user, by runing this script, you will get the terminal version interaction (it is clumsiness,but this is basic version for test)
   5. Run app.sh to get html version interaction, this is friendly for user (*￣︶￣ *).

Let me know if you have any question or any suggestions!

