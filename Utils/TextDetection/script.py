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



text="""

Wondering why everyone on your social media feed is buzzing about an ex-ball player named Bobby Bonilla who hasn't worn an MLB uniform since 2001? Or why it feels like deja vu? It's because: A. The Internet is absurd, and B. Ever since 2011, July 1 has been recognized by smart alecks across the cultural landscape (and plenty of awestruck business pundits) as #BobbyBonillaDay. 

That was the year his former employer, the New York Mets, began paying Bonilla $1,193,248.20 annually on the first of July, a sum they will continue doling out every July 1 through 2035, at which point Bonilla will be 72. Why, you ask? Because heading into the 2000 season, the Mets decided to buy out the remaining $5.9 million on their aging outfielder's contact. But rather than shell out the cash up front, team ownership accepted a deal proposed by Bonilla's agent to defer payments by a decade and disperse them over 24 years at a fixed annual interest rate of 8% (hence the funny math). 

Again — why, you ask? Because at that time, Mets ownership was infamously in thrall of late, disgraced financier Bernie Madoff, and thought they'd yield enough from short-term investments with his firm to more than compensate for future balloon payments to Bonilla. 

We all know how that worked out. And every July 1 is a continuing reminder of just how swimmingly it has all flowed for Bonilla since. But as any contemporary brand should, the Mets have tried to reclaim their fiduciary fiasco and turn it into viral gold by embracing #BobbyBonillaDay with humor — and synergy. A home run for all, indeed.

"""


if __name__ == '__main__':
    
    summ=Text_Summarization(text,size=0.3)
    s=summ.print_text()
    print(s)