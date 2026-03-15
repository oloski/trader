import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, AutoConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer

# 0. KONFIGURACJA ŚRODOWISKA
os.environ["HF_HOME"] = "/app/shared_data/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/app/shared_data/hf_cache"

# 1. PARAMETRY MODELU I DANYCH
MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
DATASET_PATH = "/app/shared_data/master_training.jsonl"
OUTPUT_DIR = "/app/shared_data/nemotron_120b_strategy_v2"

def train():
    print(f"🔄 Inicjalizacja tokenizera dla {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, 
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # 2. PATCHOWANIE KONFIGURACJI
    print(f"🛠️  Patchowanie konfiguracji modelu (MoE + NVFP4)...")
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Usuwamy quantization_config, aby uniknąć konfliktów z biblioteką transformers
    if hasattr(config, "quantization_config"):
        print("💡 Usuwanie quantization_config dla natywnej obsługi NVFP4...")
        del config.quantization_config

    # 3. ŁADOWANIE MODELU (Z poprawką ignore_mismatched_sizes)
    print(f"📥 Ładowanie modelu 120B do VRAM...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        config=config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        # KLUCZOWE: Ignorujemy różnice wymiarów w warstwach ekspertów MoE
        ignore_mismatched_sizes=True 
    )

    # 4. PRZYGOTOWANIE LORA
    # Fine-tuning 120-miliardowego mózgu wymaga zamrożenia wag bazowych
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        r=64, 
        lora_alpha=128,
        # Celujemy w kluczowe moduły Attention i Mamba
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    # 5. ŁADOWANIE TWOICH DANYCH
    print(f"📊 Ładowanie Master Datasetu: {DATASET_PATH}")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            text = (f"### Instrukcja:\n{example['instruction'][i]}\n\n"
                    f"### Dane:\n{example['input'][i]}\n\n"
                    f"### Analiza:\n{example['output'][i]}")
            output_texts.append(text)
        return output_texts

    # 6. PARAMETRY TRENINGU (Zoptymalizowane pod GB10 128GB)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=16, 
        learning_rate=1e-4,
        logging_steps=1,
        num_train_epochs=2,
        bf16=True,
        save_strategy="steps",
        save_steps=50,
        optim="paged_adamw_8bit", # Oszczędność VRAM na optymalizatorze
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        gradient_checkpointing=True,
        report_to="none"
    )

    # 7. INICJALIZACJA TRENERA
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=8192, 
        tokenizer=tokenizer,
        args=training_args,
        formatting_func=formatting_prompts_func,
    )

    print("🚀 START: Rozpoczynam trening Nemotron-3 Super 120B...")
    trainer.train()

    # 8. ZAPIS FINALNY
    trainer.save_model(OUTPUT_DIR)
    print(f"✅ Trening zakończony! Model zapisany w: {OUTPUT_DIR}")

if __name__ == "__main__":
    train()