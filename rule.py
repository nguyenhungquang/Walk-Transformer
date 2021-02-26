import pandas as pd
import networkx as nx
import itertools
import numpy as np
class Rule:
    def __init__(self,data):
        self.pd=data
        self.data=data.values
        self.data[:,[1,2]]=self.data[:,[2,1]]
        self.g=nx.MultiDiGraph()
        self.g.add_weighted_edges_from(self.data)
        self.transitivity_samples=[]
        self.transitivity_candidates={}
        self.trans_count={}
        self.trans_data={}
    def transitivity_rule_extraction(self):
        for n in self.g:
            for u,v in itertools.product(list(self.g.predecessors(n)),list(self.g.successors(n))):
                if self.g.has_edge(u,v):
                    self.transitivity_samples.append([[u,n,self.g[u][n][0]['weight']],[n,v,self.g[n][v][0]['weight']],[u,v,self.g[u][v][0]['weight']]])
    def transitivity_candidate_extract(self):
        for x,y,z in self.transitivity_samples:
            rel=(x[2],y[2],z[2])
            if rel in self.transitivity_candidates:
                self.transitivity_candidates[rel]+=1
                self.trans_data[rel].append([x,y,z])
            else:
                self.transitivity_candidates[rel]=1
                self.trans_data[rel]=[[x,y,z]]
    def score_trans(self):
        rule_list=np.array(list(self.transitivity_candidates.keys()))
        for n in self.g:
            for u,v in itertools.product(list(self.g.predecessors(n)),list(self.g.successors(n))):
                c=[self.g[u][n][0]['weight'],self.g[n][v][0]['weight']]
                b=((rule_list[:,0]==c[0])&(rule_list[:,1]==c[1]))
                if b.any():
                    coincide_rule=rule_list[np.where(b)[0]]
                    for r in coincide_rule:
                        r=tuple(r)
                        if r in self.trans_count:
                            self.trans_count[r]+=1
                        else:
                            self.trans_count[r]=1
    def transitivity_rule(self,threshold):
        self.transitivity_rule_extraction()
        self.transitivity_candidate_extract()
        self.score_trans()
        extracted_rules=[]
        for r in self.transitivity_candidates.keys():
            if self.trans_count[r]>1 and self.transitivity_candidates[r]/self.trans_count[r]>threshold:
                extracted_rules.append(r)
            else:
                self.trans_data.pop(r,None)
        return extracted_rules
