import pandas as pd
import ast
import sys
from collections import Counter
import os
import spacy
import en_core_web_lg
from collections import OrderedDict
import json
import collections
import os 
import time
import random

class BaseTask():
    def __init__(self):
        self.nlp = en_core_web_lg.load()
        self.stopwords = spacy.lang.en.stop_words.STOP_WORDS
        
    def get_POS_tags_list(self, text):
        docs = self.nlp(text)
        pos_list = []
        for word in docs:
            pos_list.append((word.text,word.pos_))

        return pos_list
    def get_enitities_list(self, text):
        docs = self.nlp(text)
        entities_dict = {word: word.label_ for word in docs.ents}
        return entities_dict
    
    def tokenise_and_lower(self, text):
        docs = self.nlp(text)
        tokens = [str(word).lower() for word in docs]
        return tokens
                        
    
    #def grab_evidence_text(self, ID):
    #    ID = ID.strip()
    #    !grep -r ID /Users/hima95/Downloads/wiki-pages-text/ >> search.txt
    #    fp = open("./search.txt",'r')
    #    text = fp.readlines()
    #    fp.close()
    #    os.remove("./search.txt")
    #    return text
    
    def extract_final_text_from_line(self, page_name, text):
        # use this function to extarct the text from a line in data-file
        try:
            page_name = " ".join(page_name.split("_"))
            if len(text)< 3:
                return " "
            common_tokens = set(self.tokenise_and_lower(text)).intersection(set(self.tokenise_and_lower(page_name)))
            if len(common_tokens) > 1:
                final_text = text
            else:
                final_text = page_name + " " + text
            return final_text
        except:
            print("Something went wrong")

    
    def longest_common_sequence(self, t1, t2):
        try:
            tkns1, tkns2 = t1.split(), t2.split()
            counter = collections.defaultdict(dict)
            for i in range(-1, len(tkns1)):
                for j in range(-1, len(tkns2)):
                    if i == -1 or j == -1:
                        counter[i][j] = 0
                    else:
                        if tkns1[i] == tkns2[j]:
                            counter[i][j] = counter[i - 1][j - 1] + 1
                        else:
                            counter[i][j] = max(counter[i - 1][j], counter[i][j - 1])
            return counter[len(tkns1) - 1][len(tkns2) - 1]
        except:
            print("C-Key Feature caliculation error")
    
class PosCountsTask(BaseTask):
    def __init__(self,df):
        self.df = df
        self.desired_pos = ["NOUN", "PROPN", "VERB", "ADJ"]
        self.grammar_pos = ["INTJ", "CCONJ", "AUX", "PUNCT","PART","SCONJ","DET","SYM", "NUM"]
        super().__init__()
        
    def make_pos_counts(self, text, pos_tag):
        docs = self.nlp(text)
        counts = Counter()
        for word in docs:
            counts[word.pos_] += 1
        # combine all counts of other grammar words than the desired list
        counts2 = Counter()
        for pos in counts.keys():
            if pos in self.grammar_pos:
                counts2["other_pos_counts"] += counts[pos]
        
        if pos_tag in counts2.keys():
            return counts2[pos_tag]
        elif pos_tag in counts.keys():
            return counts[pos_tag]
        else:
            return 0
    
    def make_counts_fields(self):
        df = self.df
        for pos in self.desired_pos:
            field_name = spacy.explain(pos) + "_counts"
            df[field_name] = df["claim"].apply(lambda x: self.make_pos_counts(x,pos))
            
        df["other_pos_counts"] = self.df["claim"].apply(lambda x: 
                                                           self.make_pos_counts(x,"other_pos_counts"))
        return df
        
class KeywordsAndEntityTask(BaseTask):
    def __init__(self,df):
        self.df = df
        self.key_pos = ["NOUN", "PROPN", "SYM", "NUM","ADJ"]
        super().__init__()
        
    def extract_keyword_list(self, claim):
        # Pick all the Entities and the key POS in the list 

        ents = self.get_enitities_list(claim)

        pos = self.get_POS_tags_list(claim)

        pos_filtered = [(key,value) for key,value in pos if value in self.key_pos]

        ents_list = list(ents.keys())
        ents_list = [str(item) for item in ents_list]
        # pos list of the entities
        pos_of_ents = self.get_POS_tags_list(" ".join(ents_list))
        pos_of_ents = [key for key,value in pos_of_ents]
        # list of all filtered pos
        pos_list = [key for key,value in pos_filtered] 
        # keywords not in entities
        other_keywords = [word for word in pos_list if word not in pos_of_ents]

        final_key_words = ents_list + other_keywords
        return final_key_words
    
    def caliculate_keywords_len(self):
        try:
            df = self.df
            df["keyword_count"] = df["claim"].apply(lambda x: len(self.extract_keyword_list(x)))
            return df 
        except:
            print("keywords caliculation error")
    
    def keywords_similarity(self, claim, candidate_sent):
        try:
            # caliculate similarity b/w claim and candidate sentence using only thier key words
            clm = self.nlp(" ".join(self.extract_keyword_list(claim)))
            evdc = self.nlp(" ".join(self.extract_keyword_list(candidate_sent)))
            return clm.similarity(evdc)
        except:
            print("C-Key Feature caliculation error")
            
    def consine_similarity(self, claim, candidate_sent):
        try:
            #caliculate similarity b/w claim and candidate sentence using only thier key words
            claim = self.nlp(claim)
            evidence = self.nlp(candidate_sent)
            clm = self.nlp(" ".join([token.text for token in claim if not token.is_stop]))
            evdc = self.nlp(" ".join([token.text for token in evidence if not token.is_stop]))
            return clm.similarity(evdc)
        except:
            print("C-Key Feature caliculation error")
        
    def common_keywords_count(self, claim, candidate_sent):
        try:
            clm = [item.lower().strip() for item in self.extract_keyword_list(claim)]
            evdc = [item.lower().strip() for item in self.extract_keyword_list(candidate_sent)]
            return len(list(set(evdc).intersection(clm)))
        except:
            print("C-Key Feature caliculation error")

    def caliculate_jacards_similarity(self, claim,evidence):
        try:
            # caliculate the jacards similarity betwen keywords of the two sentences
            claim = self.extract_keyword_list(claim)
            evidence = self.extract_keyword_list(evidence)
            intersection = set(evidence).intersection(set(claim))
            union = set(claim).union(set(evidence))
            return len(intersection)/len(union)
        except:
            print("Jacards Feature caliculation error")
    
