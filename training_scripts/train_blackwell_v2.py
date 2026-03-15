import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer
from huggingface_hub import login

# 0. KONFIGURACJA ŚRODOWISKA
# Kierujemy ogromne pliki (300GB+) na dysk współdzielony
os.environ["HF_HOME"] = "/app/shared_data/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/app/shared_data/hf_cache"

# 1. PARAMETRY MODELU I DANYCH
MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
DATASET_PATH = "/app/shared_data/master_training.jsonl"
OUTPUT_DIR = "/app/shared_data/nemotron_120b_strategy_v2"

def train():
    # Tokenizer z obsługą kodu NVIDIA
    print(f"🔄 Inicjalizacja tokenizera dla {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, 
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # 2. ŁADOWANIE MODELU (Specyficzna konfiguracja pod NVFP4 i Blackwell)
    print(f"📥 Ładowanie modelu 120B (Wymaga zainstalowanego mamba-ssm i flash-attn 2.1+)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        # Kluczowe: ustawiamy na None, aby nie wywalało błędu 'modelopt'
        # Transformers załaduje wagi bezpośrednio jako NVFP4 dzięki sterownikom NVIDIA
        quantization_config=None 
    )

    # 3. KONFIGURACJA LORA (Fine-tuning 120-miliardowego mózgu)
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        r=64, # Wysoka precyzja dla głębokiej nauki strategii
        lora_alpha=128,
        # Celujemy we wszystkie kluczowe moduły transformacji i mamby
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)

    # 4. ŁADOWANIE DANYCH
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

    # 5. PARAMETRY TRENINGU (Zoptymalizowane pod GB10 128GB)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=16, # Symulujemy batch size 16
        learning_rate=1e-4,
        logging_steps=1,
        num_train_epochs=2,
        bf16=True,                      # Natywne wsparcie Blackwella
        save_strategy="steps",
        save_steps=50,
        optim="paged_adamw_8bit",       # Oszczędność VRAM na stanach optymalizatora
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        gradient_checkpointing=True,    # Krytyczne dla oszczędności pamięci przy 8k kontekstu
        report_to="none"
    )

    # 6. INICJALIZACJA TRENERA
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=8192,           # Bardzo szerokie okno na dane rynkowe
        tokenizer=tokenizer,
        args=training_args,
        formatting_func=formatting_prompts_func,
    )

    print("🚀 START: Trening modelu Nemotron-3 Super 120B na architekturze Blackwell...")
    trainer.train()

    # 7. ZAPIS FINALNY
    trainer.save_model(OUTPUT_DIR)
    print(f"✅ Trening zakończony sukcesem. Model zapisany w: {OUTPUT_DIR}")

if __name__ == "__main__":
    train()