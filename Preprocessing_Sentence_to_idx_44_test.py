import tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
class SentenceToIdx_test():
    def sentence_to_idx(self,myobj, MAX_LEN_vector):
        freport = open("logs/report_.txt", "a", encoding="utf8")

        fraw = open("Raw_Data/raw_.txt", "w", encoding="utf8")
        flabel = open("Raw_Data/label_.txt", "w", encoding="utf8")
        fx=open("Raw_Data/x_.txt","w",encoding="utf8")
        fy = open("Raw_Data/y_.txt", "w", encoding="utf8")
        self.n_tags=myobj.n_tags
        self.max_len_vector=MAX_LEN_vector
        # Convert each sentence from list of Token to list of word_index
        my_sent=myobj.sentence
        #print(my_sent)
        #X=[]
        try:
            X = [[myobj.word2idx[w[0]] for w in s] for s in myobj.sentence]
        except Exception as err:
            print("erorr is ----"+str(err))
        fx.write(str(X))
        """for item in X:
            if len(item)>300:
                print("-----------------------------"+str(len(item)))
                print("-----------------------------" + str(item))"""
        #print(X)
        # Padding each sentence to have the same lenght
        X = pad_sequences(maxlen=self.max_len_vector, sequences=X, padding="post", value=myobj.word2idx["PAD"])

        # Convert Tag/Label to tag_index
        Y = [[myobj.tag2idx[w[1]] for w in s] for s in myobj.sentence]
        fy.write(str(Y))
        #print(X)
        # Padding each sentence to have the same lenght
        Y = pad_sequences(maxlen=self.max_len_vector, sequences=Y, padding="post", value=myobj.tag2idx["PAD"])

        # One-Hot encode
        Y = [to_categorical(i, num_classes=myobj.n_tags + 1) for i in Y]  # n_tags+1(PAD)

        #self.X_test, self.X_test, self.Y_test, self.Y_test = test_test_split(X, Y, test_size=0.9)
        self.X_test=X
        self.Y_test=Y
        print("self.X_test shape---"+str(self.X_test.shape))
        #print("self.Y_test shape---" + str(self.Y_test.shape))
        #print("self.X_test shape---" + str(self.X_test.shape))
        #print("self.Y_test shape---" + str(self.Y_test.shape))
        #return self.X_test.shape, self.X_test.shape, np.array(self.Y_test).shape, np.array(self.Y_test).shape
        #fraw.write(str(' '.join([w[0] for w in myobj.sentence[0]])))
        #print('Raw Sample: ', ' '.join([w[0] for w in myobj.sentence[0]]))
        #flabel.write(str(' '.join([w[2] for w in myobj.sentence[0]])))

        #print('Raw Label: ', ' '.join([w[2] for w in myobj.sentence[0]]))
        #print('After processing, sample:', X[0])
        #print('After processing, labels:', Y[0])
        #return self.X_test,self.X_test,self.Y_test,self.Y_test,self.n_tags,self.max_len_vector
        freport.write("self.X_test.shape---------"+str(self.X_test.shape) + "\n")
        freport.write("self.Y_test.shape---------"+str(len(self.Y_test)) + "\n")
        return self.X_test,  self.Y_test,  self.n_tags, self.max_len_vector
    def __str__(self):
        #return f'{self.X_test.shape}{self.X_test.shape}{self.Y_test.shape}{self.Y_test.shape}{self.n_tags}{self.max_len_vector}'
        return f'{self.X_test.shape}{self.Y_test.shape}{self.n_tags}{self.max_len_vector}'
    def __repr__(self):
        return f'{self.X_test}{self.Y_test}{self.n_tags}{self.max_len_vector}'

        #return f'{self.X_test}{self.X_test}{self.Y_test}{self.Y_test}{self.n_tags}{self.max_len_vector}'




