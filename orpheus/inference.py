import torch
import soundfile as sf
from unsloth import FastLanguageModel
from snac import SNAC
import os
from dotenv import load_dotenv

class OrpheusInference:
    def __init__(self, model_path, snac_model_path, output_dir="generated_samples"):
        load_dotenv()
        self.token = os.getenv("HUGGINGFACE_TOKEN")
        self.model_path = model_path
        self.snac_model_path = snac_model_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.load_models()

    def load_models(self):
        print("Loading models...")
        self.model, self.tokenizer = FastLanguageModel.for_inference(self.model_path, token=self.token)
        self.snac_model = SNAC.from_pretrained(self.snac_model_path).to("cuda")
        print("Models loaded successfully.")

    def preprocess_prompts(self, prompts, chosen_voice=None):
        print("Preprocessing prompts...")
        prompts_ = [(f"{chosen_voice}: " + p) if chosen_voice else p for p in prompts]
        all_input_ids = [self.tokenizer(p, return_tensors="pt").input_ids for p in prompts_]

        start_token = torch.tensor([[128259]], dtype=torch.int64)  # Start of human
        end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)  # End of text, End of human

        all_modified_input_ids = [
            torch.cat([start_token, input_ids, end_tokens], dim=1) for input_ids in all_input_ids
        ]

        max_length = max([ids.shape[1] for ids in all_modified_input_ids])
        all_padded_tensors = []
        all_attention_masks = []

        for modified_input_ids in all_modified_input_ids:
            padding = max_length - modified_input_ids.shape[1]
            padded_tensor = torch.cat([torch.full((1, padding), 128263, dtype=torch.int64), modified_input_ids], dim=1)
            attention_mask = torch.cat([torch.zeros((1, padding), dtype=torch.int64), torch.ones((1, modified_input_ids.shape[1]), dtype=torch.int64)], dim=1)
            all_padded_tensors.append(padded_tensor)
            all_attention_masks.append(attention_mask)

        input_ids = torch.cat(all_padded_tensors, dim=0).to("cuda")
        attention_mask = torch.cat(all_attention_masks, dim=0).to("cuda")
        print("Prompts preprocessing complete.")
        return input_ids, attention_mask

    def generate_audio(self, input_ids, attention_mask):
        print("Generating audio...")
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1200,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.1,
            num_return_sequences=1,
            eos_token_id=128258,
            use_cache=True
        )
        print("Audio generation complete.")
        return generated_ids

    def postprocess_generated_ids(self, generated_ids):
        print("Postprocessing generated IDs...")
        token_to_find = 128257
        token_to_remove = 128258
        token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)

        if len(token_indices[1]) > 0:
            last_occurrence_idx = token_indices[1][-1].item()
            cropped_tensor = generated_ids[:, last_occurrence_idx + 1:]
        else:
            cropped_tensor = generated_ids

        mask = cropped_tensor != token_to_remove
        processed_rows = [row[row != token_to_remove] for row in cropped_tensor]

        code_lists = []
        for row in processed_rows:
            row_length = row.size(0)
            new_length = (row_length // 7) * 7
            trimmed_row = row[:new_length]
            trimmed_row = [t - 128266 for t in trimmed_row]
            code_lists.append(trimmed_row)

        print("Postprocessing complete.")
        return code_lists

    def redistribute_codes(self, code_list):
        layer_1, layer_2, layer_3 = [], [], []
        for i in range((len(code_list) + 1) // 7):
            layer_1.append(code_list[7 * i])
            layer_2.append(code_list[7 * i + 1] - 4096)
            layer_3.append(code_list[7 * i + 2] - (2 * 4096))
            layer_3.append(code_list[7 * i + 3] - (3 * 4096))
            layer_2.append(code_list[7 * i + 4] - (4 * 4096))
            layer_3.append(code_list[7 * i + 5] - (5 * 4096))
            layer_3.append(code_list[7 * i + 6] - (6 * 4096))

        codes = [torch.tensor(layer_1).unsqueeze(0),
                 torch.tensor(layer_2).unsqueeze(0),
                 torch.tensor(layer_3).unsqueeze(0)]
        codes = [c.to("cuda") for c in codes]
        audio_hat = self.snac_model.decode(codes)
        return audio_hat

    def save_audio_samples(self, prompts, code_lists):
        print("Saving audio samples...")
        for i, code_list in enumerate(code_lists):
            audio_hat = self.redistribute_codes(code_list)
            audio_path = os.path.join(self.output_dir, f"sample_{i + 1}.wav")
            sf.write(audio_path, audio_hat.detach().squeeze().to("cpu").numpy(), samplerate=24000)
            print(f"Saved audio for prompt {i + 1}: {audio_path}")

    def run_inference(self, prompts, chosen_voice=None):
        input_ids, attention_mask = self.preprocess_prompts(prompts, chosen_voice)
        generated_ids = self.generate_audio(input_ids, attention_mask)
        code_lists = self.postprocess_generated_ids(generated_ids)
        self.save_audio_samples(prompts, code_lists)
        print("Inference complete.")

if __name__ == "__main__":
    prompts = [
        "ভাই, আজকে কী বার?",
    ]
    chosen_voice = None  # Single Speaker

    inference = OrpheusInference(
        model_path="path/to/fine-tuned-model", ## Replace with actual model path that you just fine-tuned
        snac_model_path="hubertsiuzdak/snac_24khz",
        output_dir="generated_samples"
    )
    inference.run_inference(prompts=prompts, chosen_voice=chosen_voice)
