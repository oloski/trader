import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer

# --- KONFIGURACJA ŚRODOWISKA ---
os.environ["HF_HOME"] = "/app/shared_data/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/app/shared_data/hf_cache"

# Token HuggingFace (wymagany dla modeli Gated)
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN, add_to_git_credential=False)
    print("✅ HuggingFace: zalogowano tokenem z env")

MODEL_NAME   = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16"  # MoE 120B/12B aktywnych, BF16
DATASET_PATH = "/app/shared_data/master_training.jsonl"
OUTPUT_DIR   = "/app/shared_data/nemotron_120b_strategy_v2"


def check_prerequisites():
    """Weryfikuje środowisko przed długim ładowaniem modelu."""
    if not torch.cuda.is_available():
        print("❌ CUDA niedostępna. Trening wymaga GPU.")
        sys.exit(1)
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU: {gpu} ({vram:.0f} GB VRAM)")

    if not os.path.exists(DATASET_PATH):
        print(f"❌ Brak datasetu: {DATASET_PATH}")
        print("   Uruchom najpierw: merge_and_shuffle.py")
        sys.exit(1)
    import subprocess
    lines = int(subprocess.check_output(["wc", "-l", DATASET_PATH]).split()[0])
    print(f"✅ Dataset: {lines:,} rekordów")

    try:
        import flash_attn  # noqa: F401
        attn = "flash_attention_2"
        print("✅ flash_attn dostępny → flash_attention_2")
    except ImportError:
        attn = "sdpa"
        print("⚠️  flash_attn niedostępny → fallback na sdpa")
    return attn


def get_target_modules(model) -> list[str]:
    """Dynamicznie wykrywa moduły liniowe modelu (Attention + Mamba SSM)."""
    from bitsandbytes.nn import Linear4bit
    linear_types = (torch.nn.Linear, Linear4bit)

    target = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_types):
            leaf = name.split(".")[-1]
            target.add(leaf)

    preferred = {"q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj",
                 "in_proj", "out_proj", "x_proj", "dt_proj"}  # Mamba SSM
    found = preferred & target
    result = list(found) if found else list(target)[:8]
    print(f"🎯 LoRA target_modules ({len(result)}): {sorted(result)}")
    return result


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = (
            f"### Instrukcja:\n{example['instruction'][i]}\n\n"
            f"### Dane:\n{example['input'][i]}\n\n"
            f"### Analiza:\n{example['output'][i]}"
        )
        output_texts.append(text)
    return output_texts


def train():
    attn_impl = check_prerequisites()

    print(f"\n🔄 Inicjalizacja tokenizera dla {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # QLoRA: model bazowy w 4-bit (~60 GB), LoRA adaptery w BF16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,  # dodatkowa kwantyzacja stałych kwantyzacji → ~3% mniej VRAM
    )

    # GB10 ma unified memory — jawnie deklarujemy ile GPU RAM jest dostępne
    vram_gb = int(torch.cuda.get_device_properties(0).total_memory / 1e9)
    max_memory = {0: f"{vram_gb - 10}GiB", "cpu": "64GiB"}
    print(f"📥 Ładowanie modelu 120B w trybie QLoRA (4-bit NF4)... [max_memory={max_memory}]")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
    )

    # Przygotowanie modelu 4-bit pod LoRA (włącza gradient checkpointing kompatybilny z bnb)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=get_target_modules(model),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print(f"\n📊 Ładowanie datasetu: {DATASET_PATH}")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

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
        optim="paged_adamw_8bit",  # QLoRA standard: oszczędza VRAM na stanach optymalizatora
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        gradient_checkpointing=False,  # już włączone przez prepare_model_for_kbit_training
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=8192,
        processing_class=tokenizer,
        args=training_args,
        formatting_func=formatting_prompts_func,
    )

    print("\n🚀 START: Trening Nemotron-3 120B (QLoRA)...")
    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    print(f"\n✅ Trening zakończony! Model zapisany w: {OUTPUT_DIR}")


if __name__ == "__main__":
    train()
