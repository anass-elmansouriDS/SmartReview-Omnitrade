from llama_cpp import Llama
import pandas as pd
import torch
import json
from scripts.config import logger

class SmartReview_Sentiment_Classifier :
    """
    Class for defining the SmartReview Sentiment Classifier.
    """
    def __init__(self,model_name="Anass-ELMANSOURI-DS/SmartReview-Sentiment-Classifier-Q4_K_M-GGUF",filename="smartreview_sentiment_classifier_gguf-q4_k_m.gguf",context_length=8192) :
        """
        Function for initializing the SmartReview Sentiment Classifier.
        @Param model_name : Name of the model to be used for inference.
        @Param filename : Filename of the model to be used for inference.
        @Param context_length : Length of the context window for the model.
        @return : None
        """       
        self.context_length = context_length
        self.model_name=model_name
        #self.hf_token=os["HF_TOKEN"]
        self.filename=filename
        self.classifier = Llama.from_pretrained(
            repo_id=self.model_name, 
            filename=self.filename, 
            n_gpu_layers=0,
            #token=hf_token, 
            n_ctx=self.context_length,
            verbose=False
        )
    def predict(self, system_prompt : str, review : str) :
        """
        Function for predicting the sentiment of a review.
        @Param system_prompt : System prompt to be used for inference.
        @Param review : Review to be classified.
        @return : Classification of the review.
        """
        prompt=system_prompt+"\nCustomer Review : "+review
        response = self.classifier.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3000
        )
        return response["choices"][0]["message"]["content"]
    
    def predict_batch(self, system_prompt : str, reviews : pd.DataFrame) :
        """
        Function for predicting the sentiment of a batch of reviews.
        @Param system_prompt (str): System prompt to be used for inference.
        @Param reviews (pd.DataFrame): Reviews to be classified.
        @return reviews_complete (pd.DataFrame): Classification of the reviews.
        """
        reviews["predicted_sentiment"]=reviews["review"].apply(lambda x : self.predict(system_prompt,x))
        mask_sent=reviews["predicted_sentiment"].isin(["Positive","Negative"])
        reviews_complete=reviews[mask_sent]
        reviews_incomplete=reviews[~mask_sent]
        while(len(reviews_incomplete)!=0) :
            logger.warning(f"Invalid classification for {len(reviews_incomplete)} reviews, reprocessing...")
            reviews_incomplete["predicted_sentiment"]=reviews_incomplete["review"].apply(lambda x : self.predict(system_prompt,x))
            reviews_complete=pd.concat([reviews_complete,reviews_incomplete[mask_sent]],axis=0)
            reviews_incomplete=reviews_incomplete[~mask_sent]
        logger.info(f"Successfully classified {len(reviews_complete)} samples.")
        return reviews_complete

class SmartReview_CC_Extractor :
    """
    Class for defining the SmartReview CC Extractor.
    """
    def __init__(self,model_name="Anass-ELMANSOURI-DS/SmartReview-CC-Extractor-Q4_K_M-GGUF",filename="smartreview-cc-extractor-q4_k_m.gguf", context_length=8192) :      
        """
        Function for initializing the SmartReview CC Extractor.
        @Param model_name (str): Name of the model to be used for inference.
        @Param filename (str): Filename of the model to be used for inference.
        @Param context_length : Length of the context window for the model.
        @return : None
        """
        self.context_length = context_length
        self.model_name=model_name
        #self.hf_token=os["HF_TOKEN"]
        self.filename=filename
        self.classifier = Llama.from_pretrained(
            repo_id=self.model_name,  
            filename=self.filename,
            n_gpu_layers=0,
            n_ctx=self.context_length,
            verbose=False
        )
    def predict(self, system_prompt : str, review : str) :
        """
        Function for extracting the compliments and complaints of a review.
        @Param system_prompt (str): System prompt to be used for inference.
        @Param review (str): Review used for C&C extraction.
        @return (str): Predictions on the review.
        """
        prompt=system_prompt+"\nCustomer Review : "+review
        response = self.classifier.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3000
        )
        return response["choices"][0]["message"]["content"]
    
    def predict_batch(self, system_prompt : str, reviews : pd.DataFrame) :
        """
        Function for extracting the compliments and complaints from a batch of reviews.
        @Param system_prompt (str): System prompt to be used for inference.
        @Param reviews (pd.DataFrame): Reviews used for C&C extraction.
        @return reviews_complete (pd.DataFrame): Predictions on the reviews.
        """
        reviews["predicted_label"]=reviews["review"].apply(lambda x : self.predict(system_prompt,x))
        mask_start=reviews["predicted_label"].str.startswith("{")
        mask_end=reviews["predicted_label"].str.endswith("}")
        mask_cmp=reviews["predicted_label"].str.contains("plaintes")
        mask_comp=reviews["predicted_label"].str.contains("compliments")
        reviews_complete=reviews[mask_start&mask_end&mask_cmp&mask_comp]
        reviews_incomplete=reviews[~(mask_start&mask_end&mask_cmp&mask_comp)]
        while(len(reviews_incomplete)!=0) :
            logger.warning(f"Invalid predictions for {len(reviews_incomplete)} reviews, reprocessing...")
            reviews_incomplete["predicted_label"]=reviews_incomplete["review"].apply(lambda x : self.predict(system_prompt,x))
            reviews_complete=pd.concat([reviews_complete,reviews_incomplete[mask_start&mask_end&mask_cmp&mask_comp]],axis=0)
            reviews_incomplete=reviews_incomplete[~(mask_start&mask_end&mask_cmp&mask_comp)]
        reviews_complete["predicted_compliments"]=reviews_complete["predicted_label"].apply(lambda x : json.loads(x)["compliments"])
        reviews_complete["predicted_complaints"]=reviews_complete["predicted_label"].apply(lambda x : json.loads(x)["plaintes"])
        logger.info(f"Successfully extracted compliments and complaints from {len(reviews_complete)} samples.")
        return reviews_complete
        
    
    
