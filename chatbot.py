
# coding: utf-8

# # Chat Bot Banco do Brasil Conta Corrente

# ### Imports

# In[ ]:


import nltk
import numpy as np
import random
import string 
import warnings

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt'); 
nltk.download('wordnet');


# ### Carregar informações

# In[2]:


f=open('faq-bb.txt','r',errors = 'ignore')
raw=f.read()
raw=raw.lower()

sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)


# ### Redução de Palavras Flexionadas

# In[3]:


lemmer = nltk.stem.RSLPStemmer() #Processo de redução de palavras flexionadas em português
def LemTokens(tokens):
    return [lemmer.stem(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# ### Cumprimetos

# In[4]:


Cumprimentos = ("olá", "oi", "ei",)
respostas = ["Oi", "Olá"]
def greeting(sentence):

    for word in sentence.split():
        if word.lower() in Cumprimentos:
            return random.choice(respostas)


# ### Processamento das Perguntas e Respostas

# In[6]:


def response(user_response):
    warnings.simplefilter('ignore')
    robo_response=''
    stopwords = nltk.corpus.stopwords.words('portuguese')
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words=stopwords)
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"Desculpa, mas não entendi."
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response


# ### BMO - BB Virtual Assisten 

# In[13]:


flag=True

print("BMO: Olá, meu nome é BMO. Posso responder suas perguntas referentes a abertura de conta no banco do Brasil.")

while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='Obrigado' or user_response=='Muito Obrigado' ):
            flag=False
            print("BMO: Não há de quer...")
        else:
            if(greeting(user_response)!=None):
                print("BMO: "+greeting(user_response))
            else:
                sent_tokens.append(user_response)
                word_tokens=word_tokens+nltk.word_tokenize(user_response)
                final_words=list(set(word_tokens))
                print("BMO: ",end="")
                print('\x1b[1;31m'+response(user_response)+'\x1b[0m')
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("BMO: Até mais!")

