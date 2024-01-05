import numpy as np
class VectorStore:
    def __init__(self):
        self.vector_data = {}
        self.vector_index = {}

    def add_vector(self,sentence,vector):
        self.vector_data[sentence] = vector
        self.update_index(sentence,vector)

    def get_vector(self,sentence):
        return self.vector_data.get(sentence)

    def update_index(self,sentence,vector):
        for existing_sentence , existing_vector in self.vector_data.items():
            similarity = (np.dot(existing_vector,vector)) / (np.linalg.norm(existing_vector))*(np.linalg.norm(vector))

            if existing_sentence not in self.vector_index:
                self.vector_index[existing_sentence] = {}
            self.vector_index[existing_sentence][sentence] = similarity    
            #vector_indes =  {exist_sentence:{sentence1:similarity with exiting vector ,
                                        #    sentence2:similarity with existing vector}}

    def find_similar_vector(self,query_vector,num_result=2):
        results = []
        for sentence , vector in self.vector_data.items():
            similarity = (np.dot(query_vector,vector)) / (np.linalg.norm(query_vector))*(np.linalg.norm(vector))
            results.append((sentence,similarity))
        results.sort(key=lambda x:x[1] , reverse=True)
        return results[:num_result]
