import argparse
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
import torch
import torchaudio.transforms as T
import os
from dotenv import load_dotenv
from snac import SNAC
import locale

class OrpheusFineTuner:
    def __init__(self, dataset_path, model_name="canopylabs/orpheus-3b-0.1-pretrained", output_dir="outputs"):
        print("Initializing Orpheus Fine-Tuner...")
        load_dotenv()
        self.token = os.getenv("HUGGINGFACE_TOKEN")
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.output_dir = output_dir
        self.setup_constants()
        locale.getpreferredencoding = lambda: "UTF-8"
        print("Initialization complete.")

    def setup_constants(self):
        print("Setting up constants...")
        self.tokeniser_length = 128256
        self.start_of_text = 128000
        self.end_of_text = 128009
        self.start_of_speech = self.tokeniser_length + 1
        self.end_of_speech = self.tokeniser_length + 2
        self.start_of_human = self.tokeniser_length + 3
        self.end_of_human = self.tokeniser_length + 4
        self.start_of_ai = self.tokeniser_length + 5
        self.end_of_ai = self.tokeniser_length + 6
        self.pad_token = self.tokeniser_length + 7
        self.audio_tokens_start = self.tokeniser_length + 10
        print("Constants setup complete.")

    def load_model(self):
        print("Loading model...")
        dtype = None
        load_in_4bit = True # If think for  your use case, you can set it to false. 
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=2048,
            dtype=dtype,
            load_in_4bit=load_in_4bit, 
            token=self.token,
        )
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
            lora_alpha=64, # I think you can go higher here.
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        print("Model loaded successfully.")

    def prepare_snac_model(self):
        print("Loading SNAC model...")
        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
        self.snac_model = self.snac_model.to("cuda")
        print("SNAC model loaded successfully.")

    def tokenise_audio(self, waveform):
        print("Tokenizing audio...")
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        waveform = waveform.to(dtype=torch.float32)
        resample_transform = T.Resample(orig_freq=self.ds_sample_rate, new_freq=24000)
        waveform = resample_transform(waveform)
        waveform = waveform.unsqueeze(0).to("cuda")
        
        with torch.inference_mode():
            codes = self.snac_model.encode(waveform)

        all_codes = []
        for i in range(codes[0].shape[1]):
            all_codes.extend([
                codes[0][0][i].item()+128266,
                codes[1][0][2*i].item()+128266+4096,
                codes[2][0][4*i].item()+128266+(2*4096),
                codes[2][0][(4*i)+1].item()+128266+(3*4096),
                codes[1][0][(2*i)+1].item()+128266+(4*4096),
                codes[2][0][(4*i)+2].item()+128266+(5*4096),
                codes[2][0][(4*i)+3].item()+128266+(6*4096)
            ])
        print("Audio tokenization complete.")
        return all_codes

    def pad_sequences(self, sequences, pad_token):
        print("Padding sequences...")
        max_length = max(len(seq) for seq in sequences)
        padded_sequences = [
            seq + [pad_token] * (max_length - len(seq)) for seq in sequences
        ]
        print("Padding complete.")
        return padded_sequences

    def prepare_dataset(self):
        print("Preparing dataset...")
        # -------------------Start Change---------------------
        #  Directly corelate to `SUST-CSE-Speech/banspeech` format.
        #  You need to update this for your dataset Format.
        dataset = load_dataset(self.dataset_path, split="audio_books")
        dataset = dataset.remove_columns(["file_path"])
        dataset = dataset.rename_column("transcription", "text")
        self.ds_sample_rate = dataset[0]["audio"]["sampling_rate"]
        #--------------------End Change--------------------
        
        
        dataset = dataset.map(self.add_codes, remove_columns=["audio"])
        dataset = dataset.filter(lambda x: x["codes_list"] is not None and len(x["codes_list"]) > 0)
        dataset = dataset.map(self.remove_duplicate_frames)
        dataset = dataset.map(self.create_input_ids, remove_columns=["text", "codes_list"])
        
        # Padding logic [I did not test this. Please comment it out in it doesn't work]
        def pad_batch(batch):
            batch["input_ids"] = self.pad_sequences(batch["input_ids"], self.pad_token)
            batch["labels"] = self.pad_sequences(batch["labels"], self.pad_token)
            batch["attention_mask"] = self.pad_sequences(batch["attention_mask"], 0)
            return batch

        dataset = dataset.map(pad_batch, batched=True)

        columns_to_keep = ["input_ids", "labels", "attention_mask"]
        columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]
        self.dataset = dataset.remove_columns(columns_to_remove)
        print("Dataset preparation complete.")

    def add_codes(self, example):
        codes_list = None
        try:
            answer_audio = example.get("audio")
            if answer_audio and "array" in answer_audio:
                codes_list = self.tokenise_audio(answer_audio["array"])
        except Exception as e:
            print(f"Skipping row due to error: {e}")
        example["codes_list"] = codes_list
        return example

    def remove_duplicate_frames(self, example):
        vals = example["codes_list"]
        if len(vals) % 7 != 0:
            raise ValueError("Input list length must be divisible by 7")

        result = vals[:7]
        for i in range(7, len(vals), 7):
            if vals[i] != result[-7]:
                result.extend(vals[i:i+7])

        example["codes_list"] = result
        return example

    def create_input_ids(self, example):
        text_prompt = f"{example['source']}: {example['text']}" if "source" in example else example["text"]
        text_ids = self.tokenizer.encode(text_prompt, add_special_tokens=True)
        text_ids.append(self.end_of_text)

        example["text_tokens"] = text_ids
        input_ids = (
            [self.start_of_human]
            + example["text_tokens"]
            + [self.end_of_human]
            + [self.start_of_ai]
            + [self.start_of_speech]
            + example["codes_list"]
            + [self.end_of_speech]
            + [self.end_of_ai]
        )
        example["input_ids"] = input_ids
        example["labels"] = input_ids
        example["attention_mask"] = [1] * len(input_ids)
        return example

    def train(self):
        print("Starting training...")
        training_args = TrainingArguments(
            per_device_train_batch_size=1, # If you have H100, go crazy! Gotta  make sure padding is working.
            gradient_accumulation_steps=4,
            save_strategy="steps",
            save_steps=100,
            warmup_steps=5,
            num_train_epochs=10,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=self.output_dir,
            # report_to="wandb", # Uncomment if you want to use wandb for logging
            # run_name="Orpheus-bn-Fine-Tune",
        )

        trainer = Trainer(
            model=self.model,
            train_dataset=self.dataset,
            args=training_args,
        )
        
        trainer_stats = trainer.train()
        print("Training complete.")
        return trainer_stats

    def save_model(self, output_path="Orpheus-bangla"):
        print(f"Saving model to {output_path}...")
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print("Model saved successfully.")

    def marge_model(self, output_path="Orpheus-bangla-merged"):
        print("Merging model...")
        try:
            if hasattr(self.model, "save_pretrained_merged"):
                self.model.save_pretrained_merged(
                    output_path, 
                    self.tokenizer, 
                    save_method="merged_16bit"
                )
                print(f"Model merged and saved successfully to {output_path}.")
            else:
                print("Error: The model does not support merging weights.")
        except Exception as e:
            print(f"An error occurred during model merging: {e}")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Orpheus model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for checkpoints")
    parser.add_argument("--model_path", type=str, default="canopylabs/orpheus-3b-0.1-pretrained", 
                        help="Path to the pre-trained model")
    args = parser.parse_args()

    fine_tuner = OrpheusFineTuner(
        dataset_path=args.dataset_path,
        model_name=args.model_path,
        output_dir=args.output_dir
    )
    
    fine_tuner.load_model()
    fine_tuner.prepare_snac_model()
    fine_tuner.prepare_dataset()
    fine_tuner.train()
    fine_tuner.save_model()
    fine_tuner.marge_model()
    print("Fine-tuning complete. Model saved to:", args.output_dir)

if __name__ == "__main__":
    main()