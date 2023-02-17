# SemEval23_LegalEval_TeamLRL_NC

Repository for submission of Team LRL_NC at SemEval23-Task6:Rhetorical Role Prediction for understanding Legal Texts

The task website is given below:
https://sites.google.com/view/legaleval/home

The dataset is available at https://github.com/Legal-NLP-EkStep/rhetorical-role-baseline

The additional ILDC Corpus can be requested at the link, https://github.com/Exploration-Lab/CJPE

For using the given codes, Python 3.8.10 is used and the packages used are given in `requirements.txt`.

The code in file `data_prep.py` prepares some necessary files required for further processing.

The ILDC dataset can be used for further pretraining RoBERTa on MLM task, using the code `finetune_roberta_MLM.py`

The HDP features can be prepared using `hdp_features.py`

The final model can be trained using `proposed_system.py`
