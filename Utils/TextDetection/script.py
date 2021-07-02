import spacy
from  spacy.lang.en.stop_words import  STOP_WORDS

from heapq import nlargest

class Text_Summarization:
    


    stopWords=list(STOP_WORDS)
    nlp=spacy.load('en_core_web_sm')
    def __init__(self,doc,size=0.3):


        self.doc=self.nlp(doc)
        self.size=size

    def get_punctuation(self):
        from string import punctuation
        punctuation=list(punctuation)
        punctuation.append('\n')

        return punctuation

    
    def tokenize(self):
        
        docs=self.doc
        tokens=[token.text for token in docs]


        punctuation=self.get_punctuation()
        tokens=[ token for token in tokens if token not in punctuation ]

        return tokens

    def word_frequencey_couter(self):
        punctuation=self.get_punctuation()
        word_frequency={}
        for word in self.doc:
            if word.text.lower() not in self.stopWords:
                if (word.text.lower() not in punctuation ):
                    if word.text not in word_frequency.keys():
                        word_frequency[word.text] = 1
                    else:
                        word_frequency[word.text] +=1
        
        max_frequency=max(word_frequency.values())    

        for word in word_frequency.keys():
            word_frequency[word]=word_frequency[word]/max_frequency
        
        return word_frequency,max_frequency
    

    def sentence_tokenize(self):
        sentence_tokens=[sent for sent in self.doc.sents]
        return sentence_tokens
    
    def sentence_tokenize_score(self):
        word_frequency,_=self.word_frequencey_couter()
        sentence_tokens=self.sentence_tokenize()
    
        sentence_score={}
        
        for sent in sentence_tokens:
            for word in sent:
                if (word.text.lower() in word_frequency.keys()):
                    if(sent not in sentence_score.keys()):
                        sentence_score[sent]=word_frequency[word.text.lower()]
                    else:
                        sentence_score[sent] +=word_frequency[word.text.lower()]
        return sentence_score
    
    def summarize_text(self):
        sentence_score= self.sentence_tokenize_score()
        sentence_tokens=self.sentence_tokenize()
        select_length=int(len(sentence_tokens) * self.size)
        # print(self.size)
        get_summary=nlargest(select_length,sentence_score,key=sentence_score.get)
        # print(get_summary)

        final_summary=[word.text for word in get_summary]
        summary=' '.join(final_summary)

        return summary

        
    def print_text(self):
        i=self.summarize_text()
        return i






if __name__ == '__main__':
    
    summ=Text_Summarization(text,size=0.3)
    s=summ.print_text()
    print(s)