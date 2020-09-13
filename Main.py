# TENSOR 2.0.0
from Loadrawdata_show_2 import Loadrawdata_show_train
from Word_tags_to_idx_3 import WordTagsToIdx_train
from Preprocessing_Sentence_to_idx_4 import SentenceToIdx_train
##############################################


filename_train='Raw_Data/Test_set.txt'
max_len_input=300
loadrawdata_train=Loadrawdata_show_train()
wrd_tag_vec_train=WordTagsToIdx_train()
wrd_tag_vec_train.word_tags_to_idx(loadrawdata_train.load(filename_train),max_len_input)
sent_idx_train=SentenceToIdx_train()
sent_idx_train.sentence_to_idx(wrd_tag_vec_train,max_len_input)
