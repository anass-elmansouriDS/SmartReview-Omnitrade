from models import SmartReview_Sentiment_Classifier, SmartReview_CC_Extractor
from sklearn.metrics import precision_recall_f_score_support
import mlflow

def evaluate_sentiment_classification(model_name : str, revision : str, system_prompt : str, test_dataset) :
    classifier=SmartReview_Sentiment_Classifier(model_name, revision)
    preds=classifier.predict_batch(system_prompt,test_dataset)
    preds["encoded_predicted_sentiment"]=preds['sentiment'].apply(lambda x  : 1 if x=="Positive" else 0)
    overall_scores=precision_recall_f_score_support(preds['encoded_sentiment'], preds["encoded_predicted_sentiment"], average='weighted')
    per_class_scores=precision_recall_f_score_support(preds['encoded_sentiment'], preds["encoded_predicted_sentiment"])
    return {"Overall" : overall_scores, "Per Class" : per_class_scores}

def log_sentiment_model_performance(model_name : str, revision : str, system_prompt : str, dataset_path : str, overall_scores, per_class_scores) :
    mlflow.set_tracking_uri("databricks")
    experiment_id=mlflow.set_experiment("/Workspace/Users/anas.elmansouri040@gmail.com/SmartReview-Modeling")
    with mlflow.start_run(run_name=f"Model name : {model_name}, Revision : {revision}") as run:
        mlflow.log_artifact(dataset_path)
        mlflow.log_param("revision", revision)
        mlflow.log_param("features_used",["review"])
        mlflow.log_param("label_used", "sentiment")
        mlflow.log_param("classes", ["Positive","Negative"])
        mlflow.log_param("Prompt",system_prompt)
        mlflow.log_metric("overall_precision", overall_scores[0])
        mlflow.log_metric("overall_recall", overall_scores[1])
        mlflow.log_metric("overall_F1", overall_scores[2])
        mlflow.log_metric("Positive_class_precision",per_class_scores[0][1])
        mlflow.log_metric("Positive_class_recall",per_class_scores[1][1])
        mlflow.log_metric("Positive_class_F1",per_class_scores[2][1])
        mlflow.log_metric("Negative_class_precision",per_class_scores[0][0])
        mlflow.log_metric("Negative_class_recall",per_class_scores[1][0])
        mlflow.log_metric("Negative_class_F1",per_class_scores[2][0])

def evaluate_cc_extraction(model_name : str, revision : str, system_prompt : str, test_dataset) :
    pass

def log_cc_model_performance(model_name : str, revision : str, system_prompt : str, eval_system_prompt : str, eval_name : str, dataset_path : str, metrics) :
    mlflow.set_tracking_uri("databricks")
    experiment_id=mlflow.set_experiment("/Workspace/Users/anas.elmansouri040@gmail.com/SmartReview-Modeling")
    with mlflow.start_run(run_name=f"Model name : {model_name}, Revision : {revision}") as run:
        mlflow.log_artifact(dataset_path)
        mlflow.log_param("features_used",["review"])
        mlflow.log_param("label_used", "label")
        mlflow.log_param("System prompt",system_prompt)
        mlflow.log_param("LLM judge system prompt", eval_system_prompt)
        mlflow.log_param("LLM judge model name", eval_name)
        mlflow.log_metric("Complaints Extraction Score", metrics[0])
        mlflow.log_metric("Compliments Extraction Score", metrics[1])
        mlflow.log_metric("Critical Fail Rate on Complaints", metrics[2])
        mlflow.log_metric("Critical Fail Rate on Compliments", metrics[3])
        mlflow.log_metric("Average CFR", metrics[4])
        mlflow.log_metric("Average Score", metrics[5])
