from datasets import Dataset
from pyspark.sql import SparkSession
from scripts.config import logger

def get_dataset(train:bool=False,sentiment:bool=False) :
    """
    Function to fetch the dataset for training or inference.
        @Param: train (bool): Flag to indicate if the dataset is for training or inference, Default=False.
        @Param: sentiment (bool): Flag to indicate if the dataset is for sentiment or C&C extraction, Default=False.
        @return: Dataset (spark.DataFrame): Spark Dataframe containing the dataset for training or inference.
    """
    spark = SparkSession.builder.getOrCreate()
    dataset=spark.read.table("workspace.default.omnitrade_customer_reviews")
    if sentiment :
        if train :
            logger.info(f"Fetching sentiment training dataset...")
            return dataset.select(["review_id","review","sentiment"]).where(dataset["sentiment"].isNotNull())
        else :
            logger.info(f"Fetching sentiment inference dataset...")
            return dataset.where(dataset["predicted_sentiment"].isNull()).select(["review_id","review","predicted_sentiment"])
    else :
        if train :
            logger.info(f"Fetching C&C extraction training dataset...")
            return dataset.select(["review_id","review","label"]).where(dataset["label"].isNotNull())
        else :
            logger.info(f"Fetching C&C extraction inference dataset...")
            return dataset.where(dataset["predicted_compliments"].isNull()).select(["review_id","review","predicted_compliments","predicted_complaints"])

def formatting_prompts_func(example: dict, system_prompt:str, sentiment:bool=False) :
    """
    Function to format the dataset for training with unsloth.
        @Param: example (dict): Dictionary containing the example data.
        @Param: system_prompt (str): System prompt to be used for formatting the dataset.
        @Param: sentiment (bool): Flag to indicate if the dataset is for sentiment or C&C extraction, Default=False
        @returns: dict: Dictionary containing the formatted example data.
    """
    if sentiment :
        return {
            "messages": [
                {"role": "user", "content": system_prompt + example["review"]},
                {"role": "assistant", "content": example["sentiment"]}
            ]
        }
    else :
        return {
            "messages": [
                {"role": "user", "content": system_prompt + example["review"]},
                {"role": "assistant", "content": example["label"]}
            ]
        }

def format_conversation(messages : list, eos_token : str):
    """
    Function to format the dataset as conversations for training with unsloth.
        @Param: messages (list): List of messages to be formatted.
        @Param: eos_token (str): End of sequence token to be used for formatting the dataset.
        @returns: str: Formatted dataset as conversations.
    """
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        formatted += f"<|{role}|>\n{content}"
    formatted += eos_token
    return formatted
    
def prepare_training_dataset(system_prompt : str, eos_token : str, sentiment=False) :
    """
    Function to prepare the training dataset for training with unsloth.
        @Param: system_prompt (str): System prompt to be used for formatting the dataset.
        @Param: eos_token (str): End of sequence token to be used for formatting the dataset.
        @Param: sentiment (bool): Flag to indicate if the dataset is for sentiment or C&C extraction, Default=False.
        @returns: training_dataset (datasets.Dataset): Training dataset formatted for training with unsloth.
    """
    dataset=get_dataset(train=True,sentiment=sentiment).toPandas()
    dataset.attrs.clear()
    training_dataset=Dataset.from_pandas(dataset)
    try :
        logger.info("Formatting training dataset to match conversational format...")
        training_dataset=training_dataset.map(formatting_prompts_func,fn_kwargs={"system_prompt": system_prompt,"sentiment": sentiment})
        training_dataset=training_dataset.map(lambda x: {"text": format_conversation(x["messages"],eos_token)})
        logger.info("Successfully Formatted training dataset to match conversational format.")
    except Exception as e :
        logger.critical(f"Error formatting training dataset to match conversational format with exception {e}.")
        raise e
    return training_dataset


