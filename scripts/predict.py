from scripts.data import get_dataset
from scripts.utils import str2bool
from scripts.config import logger
from scripts.models import SmartReview_Sentiment_Classifier, SmartReview_CC_Extractor
from pyspark.sql import SparkSession
import argparse

def parse_args_predict():
    """
    Function for parsing arguments for the predict function.

    @Return: The parsed arguments.
    """ 
    system_prompt_sentiment="""You are a Customer Review Analyst that predicts the sentiment of a customer's review concerning a product.
Customer reviews will come in one of four languages/dialects: English, French, Arabic written Moroccan darija, and Latin written moroccan darija.
You must choose between one of the following labels for each review: ["Positive","Negative"].
Only respond with the label name and nothing else.
"""
    system_prompt_cc="""You are a customer review analyst whose job is to extract customer complaints and compliments from the input customer review.
Return the complaints and compliments in a string written in French shaped like this: {"plaintes" : [(list of complaints you extracted seperated by a comma)], "compliments" : [(list of compliments you extracted seperated by a comma)]}.​
Always output only the string and do not wrap it in JSON markdown or add any extra text before or after the string.​
Always write the compliments and complaints in a professional and formal register in French.​

Additional rules:
Always extract only complaints or compliments that are specific to the main product being reviewed; ignore any compliments or complaints that target another product, even if it is from the same brand as the main product.​
Always include any compliments or complaints related to the delivery or shipping of the product (speed, packaging, condition on arrival, courier experience, etc.), because delivery is considered an important part of the overall product experience.​
If the review contains no relevant complaint or compliment about the main product or its delivery, return empty lists: {"plaintes" : [], "compliments" : []}
"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentiment", required=True, type=str2bool)
    parser.add_argument("--model_repo_sentiment", default="Anass-ELMANSOURI-DS/SmartReview-Sentiment-Classifier-Q4_K_M-GGUF")
    parser.add_argument("--model_filename_sentiment", default="smartreview_sentiment_classifier_gguf-q4_k_m.gguf")
    parser.add_argument("--model_repo_cc", default="Anass-ELMANSOURI-DS/SmartReview-CC-Extractor-Q4_K_M-GGUF")
    parser.add_argument("--model_filename_cc", default="smartreview-cc-extractor-q4_k_m.gguf")
    parser.add_argument("--context_length", default=8192)
    parser.add_argument("--system_prompt_sentiment", default=system_prompt_sentiment)
    parser.add_argument("--system_prompt_cc", default=system_prompt_cc)
    return parser.parse_args()

def main() :
    """
    Function for predicting the sentiment of a customer's review and extracting customer complaints and compliments from the input customer review.
    
    @Return: None
    """ 
    try :
        logger.info("Spark session creation started...")
        spark = SparkSession.builder.getOrCreate()
        logger.info("Spark session created.")
    except Exception as e:
        logger.critical(f"Spark session creation failed with exception : {e}.")
        raise e
    args=parse_args_predict()
    try :
        dataset = get_dataset(sentiment=args.sentiment).toPandas()[:10]
        logger.info("Dataset was successfully fetched and is now ready for inference.")
    except Exception as e:
        logger.critical(f"Fetching dataset failed with exception : {e}.")
        raise e
    if args.sentiment:
        try :
            logger.info("Loading sentiment classifier...")
            model = SmartReview_Sentiment_Classifier(model_name=args.model_repo_sentiment, filename=args.model_filename_sentiment, context_length=args.context_length)
            logger.info("Sentiment classifier was successfully loaded and now ready for inference.")
        except Exception as e:
            logger.critical(f"Sentiment classifier failed to load with exception : {e}.")
            raise e
        try :
            logger.info(f"Inference started for {len(dataset)} samples...")
            preds=model.predict_batch(system_prompt=args.system_prompt_sentiment, reviews=dataset)
        except Exception as e :
            logger.critical(f"Inference failed with exception : {e}.")
            raise e
        try :
            logger.info(f"Saving inference results for {len(dataset)} samples to table : workspace.default.smartreview_sentiment_preds...")
            sdf_preds=spark.createDataFrame(preds)
            sdf_preds.write.mode("append").saveAsTable("workspace.default.smartreview_sentiment_preds")
            logger.info(f"Successfully saved inference results for {len(dataset)} samples to table : workspace.default.smartreview_sentiment_preds...")
        except Exception as e :
            logger.critical(f"Saving inference results failed with exception : {e}.")
            raise e
    else :
        try :
            logger.info("Loading C&C Extractor...")
            model= SmartReview_CC_Extractor(model_name=args.model_repo_cc, filename=args.model_filename_cc, context_length=args.context_length)
            logger.info("C&C Extractor was successfully loaded and now ready for inference.")
        except Exception as e:
            logger.critical(f"C&C Extractor failed to load with exception : {e}.")
            raise e
        try :
            logger.info(f"Inference started for {len(dataset)} samples...")
            preds=model.predict_batch(system_prompt=args.system_prompt_cc, reviews=dataset)
            logger.info(f"Inference succeded for {len(dataset)} samples.")
        except Exception as e :
            logger.critical(f"Inference failed with exception : {e}.")
            raise e
        try :
            logger.info(f"Saving inference results for {len(dataset)} samples to table : workspace.default.smartreview_CC_preds...")
            sdf_preds=spark.createDataFrame(preds)
            sdf_preds.write.mode("append").saveAsTable("workspace.default.smartreview_CC_preds")
            logger.info(f"Successfully saved inference results for {len(dataset)} samples to table : workspace.default.smartreview_CC_preds...")
        except Exception as e :
            logger.critical(f"Saving inference results failed with exception : {e}.")
            raise e  

if __name__=="__main__" :
    main()