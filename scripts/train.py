from scripts.data import prepare_training_dataset
from scripts.utils import str2bool
from unsloth import FastLanguageModel
import torch
from scripts.config import logger
from trl import SFTConfig, SFTTrainer

def load_model(model_name : str, max_seq_length : int, dtype : str, load_in_4bit=True) :
    """
    Function for loading the model and tokenizer.

    @Param model_name (str): The name of the model to load.
    @Param max_seq_length (int): The maximum sequence length for the model.
    @Param dtype (str): The data type for the model.
    @Param load_in_4bit (bool): Whether to load the model in 4bit mode.

    @Return: The loaded model and tokenizer.
    """ 
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    return model, tokenizer

def setup_lora_adapters(model, r=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj",], lora_alpha=16, lora_dropout=0, bias="none", use_rslora=False, loftq_config= None, random_state=42) :
    """
    Function for setting up the LoRA adapters for the model.

    @Param model (unsloth.FastLanguageModel): The model to set up the LoRA adapters for.
    @Param r (int): The rank of the LoRA adapters.
    @Param target_modules (list): The target modules for the LoRA adapters.
    @Param lora_alpha (int): The alpha value for the LoRA adapters.
    @Param lora_dropout (float): The dropout value for the LoRA adapters.
    @Param bias (str): The bias value for the LoRA adapters.
    @Param use_rslora (bool): Whether to use RSLORA adapters.
    @Param loftq_config (dict): The LoRA configuration for the LoRA adapters.
    @Param random_state (int): The random state for the LoRA adapters.

    @Return: The model with the LoRA adapters set up.
    """
    model = FastLanguageModel.get_peft_model(
        model,
        r = r,
        target_modules = target_modules,
        lora_alpha = lora_alpha,
        lora_dropout = lora_dropout,
        bias = bias,    
        use_gradient_checkpointing = "unsloth",
        random_state = random_state,
        use_rslora = False, 
        loftq_config = None,
    )
    return model

def setup_training_args(output_dir : str, assistant_only_loss=True, per_device_train_batch_size=10, gradient_accumulation_steps=4, warmup_steps=5, num_train_epochs=3, max_steps=150, learning_rate=2e-4, logging_steps=5, save_strategy="steps", save_steps=20, save_total_limit=3, optim="adamw_8bit", weight_decay=0.001, lr_scheduler_type="linear", seed=3407, report_to="none") :
    """
    Function for setting up the training arguments for the model.

    @Param assistant_only_loss (bool): Whether to use assistant only loss.
    @Param per_device_train_batch_size (int): The batch size for training.
    @Param gradient_accumulation_steps (int): The number of gradient accumulation steps.
    @Param warmup_steps (int): The number of warmup steps.
    @Param num_train_epochs (int): The number of training epochs.
    @Param max_steps (int): The maximum number of steps.
    @Param learning_rate (float): The learning rate for training.
    @Param logging_steps (int): The number of logging steps.
    @Param save_strategy (str): The save strategy for training.
    @Param save_steps (int): The number of save steps.
    @Param save_total_limit (int): The total limit for saving.
    @Param optim (str): The optimizer for training.
    @Param weight_decay (float): The weight decay for training.
    @Param lr_scheduler_type (str): The learning rate scheduler type for training.
    @Param seed (int): The random seed for training.
    @Param output_dir (str): The output directory for training.
    @Param report_to (str): The reporting tool for training.

    @Return args (transformers.TrainingArguments): The training arguments for the model.
    """
    args=SFTConfig(
        assistant_only_loss= assistant_only_loss,
        per_device_train_batch_size = per_device_train_batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        warmup_steps = warmup_steps,
        num_train_epochs = num_train_epochs, # Set this for 1 full training run.
        max_steps = max_steps,
        learning_rate = learning_rate,
        logging_steps = logging_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        optim = optim,
        weight_decay = weight_decay,
        lr_scheduler_type = lr_scheduler_type,
        seed = seed,
        output_dir = output_dir,
        report_to = report_to,
    )
    return args

def setup_training(model, tokenizer, train_dataset, max_seq_length, args, packing=False) :
    """
    Function for setting up the training for the model.

    @Param model (unsloth.FastLanguageModel): The model to train.
    @Param tokenizer (unsloth.FastLanguageTokenizer): The tokenizer for the model.
    @Param train_dataset (torch.utils.data.Dataset): The training dataset for the model.
    @Param max_seq_length (int): The maximum sequence length for the model.
    @Param packing (bool): Whether to pack the data for training.
    @Param args (transformers.TrainingArguments): The training arguments for the model.

    @Return trainer (transformers.SFTTrainer): The trainer for the model.
    """
    trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    packing = packing,
    args = args
    )
    return trainer

def parse_args_train():
    """
    Function for parsing arguments from the CLI to use in the code.

    @Param: None

    @Return: args (parser.parse_args): The arguments from the CLI.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentiment", required=True, type=str2bool)
    parser.add_argument("--model_name", default="Anass-ELMANSOURI-DS/SmartReview-Sentiment-Classifier-Q4_K_M-GGUF")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--dtype", default="none")
    parser.add_argument("--max_seq_length", default=8192)
    parser.add_argument("--system_prompt", required=True)
    parser.add_argument("--assistant_only_loss", default=True)
    parser.add_argument("--per_device_train_batch_size", required=True)
    parser.add_argument("--gradient_accumulation_steps", required=True)
    parser.add_argument("--warmup_steps", default=5)
    parser.add_argument("--num_train_epochs", required=True)
    parser.add_argument("--max_steps", default=150)
    parser.add_argument("--learning_rate", default=2e-4)
    parser.add_argument("--logging_steps", default=5)
    parser.add_argument("--save_strategy", default="steps")
    parser.add_argument("--save_steps", default=20)
    parser.add_argument("--save_total_limit", default=3)
    parser.add_argument("--optim", default="adamw_8bit")
    parser.add_argument("--weight_decay", default=0.001)
    parser.add_argument("--lr_scheduler_type", default="linear")
    parser.add_argument("--seed", default=3407)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--report_to", default="none")
    return parser.parse_args()

def main() :
    args=parse_args_train()
    try :
        logger.info("Loading model...")
        model, tokenizer = load_model(args.model_name, args.max_seq_length, args.dtype)
        logger.info("Model loaded.")
    except Exception as e :
        logger.critical(f"Error loading model: {e}")
        raise e
    try :
        logger.info("Applying lora adapters to the model...")
        model = setup_lora_adapters(model)
        logger.info("Successfully applied lora adapters to the model.")
    except Exception as e :
        logger.critical(f"Error applying lora adapters to the model: {e}")
        raise e
    eos_token=tokenizer.eos_token
    try:
        logger.info("Loading training data...")
        training_data = prepare_training_data(sentiment=args.sentiment, system_prompt=args.system_prompt, eos_token=eos_token)
        logger.info("Successfully loaded the training data.")
    except Exception as e :
        logger.critical(f"Failed to load training data with exception : {e}")
        raise e
    try :
        logger.info("Setting up the training arguments for unsloth...")
        args = setup_training_args(assistant_only_loss=args.assistant_only_loss, per_device_train_batch_size=args.per_device_train_batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps, warmup_steps=args.warmup_steps, num_train_epochs=args.num_train_epochs, max_steps=args.max_steps, learning_rate=args.learning_rate, logging_steps=args.logging_steps, save_strategy=args.save_strategy, save_steps=args.save_steps, save_total_limit=args.save_total_limit, optim=args.optim, weight_decay=args.weight_decay, lr_scheduler_type=args.lr_scheduler_type, seed=args.seed, output_dir=args.output_dir, report_to=args.report_to)
        logger.info("Training arguments set and ready for training.")
    except Exception as e :
        logger.critical("Failed to setup training arguments with exception : {e}")
        raise e
    try :
        logger.info("Starting training...")
        trainer = setup_training(model, tokenizer, training_data, max_seq_length, args)
        results=trainer.train()
        logger.info(f"Starting finished with training results {results}...")
    except Exception as e :
        logger.critical(f"Training failed with exception : {e}")
        raise e
    hf_token=OS["HF_TOKEN"]
    if sentiment :
        trainer.model.push_to_hub("Anass-kirito-2001/SmartReview-Sentiment-Classifier", use_auth_token=hf_token,revision=revision)
        trainer.tokenizer.push_to_hub("Anass-kirito-2001/SmartReview-Sentiment-Classifier", use_auth_token=hf_token,revision=revision)
    else :
        trainer.model.push_to_hub("Anass-ELMANSOURI-DS/SmartReview-C_C-Extraction-Model", use_auth_token=hf_token,revision=revision)
        trainer.tokenizer.push_to_hub("Anass-ELMANSOURI-DS/SmartReview-C_C-Extraction-Model", use_auth_token=hf_token,revision=revision)



    

    


