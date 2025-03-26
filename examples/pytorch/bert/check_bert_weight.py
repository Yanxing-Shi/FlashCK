import transformers

from transformers import BertTokenizerFast, BertModel


model_id="google-bert/bert-base-uncased" 


tokenizer = BertTokenizerFast.from_pretrained(model_id)
model = BertModel.from_pretrained(model_id)
# print(vars(model.config))
for key, value in model.state_dict().items():
    print(key, value.shape)



