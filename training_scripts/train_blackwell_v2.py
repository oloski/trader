import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, AutoConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer

# --- KONFIGURACJA ŚRODOWISKA ---
os.environ["HF_HOME"] = "/app/shared_data/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/app/shared_data/hf_cache"

MODEL_NAME   = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
DATASET_PATH = "/app/shared_data/master_training.jsonl"
OUTPUT_DIR   = "/app/shared_data/nemotron_120b_strategy_v2"


def check_prerequisites():
    """Weryfikuje środowisko przed długim ładowaniem modelu."""
    # Fix 5: GPU
    if not torch.cuda.is_available():
        print("❌ CUDA niedostępna. Trening wymaga GPU.")
        sys.exit(1)
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU: {gpu} ({vram:.0f} GB VRAM)")

    # Fix 5: dataset
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Brak datasetu: {DATASET_PATH}")
        print("   Uruchom najpierw: merge_and_shuffle.py")
        sys.exit(1)
    import subprocess
    lines = int(subprocess.check_output(["wc", "-l", DATASET_PATH]).split()[0])
    print(f"✅ Dataset: {lines:,} rekordów")

    # Fix 1: flash_attn
    try:
        import flash_attn  # noqa: F401
        attn = "flash_attention_2"
        print("✅ flash_attn dostępny → flash_attention_2")
    except ImportError:
        attn = "sdpa"
        print("⚠️  flash_attn niedostępny → fallback na sdpa (wolniejsze)")
    return attn


def get_target_modules(model) -> list[str]:
    """
    Fix 3: Dynamicznie wykrywa moduły liniowe modelu zamiast hardcodu.
    Dla Nemotron-3 (MoE + Mamba) nazwy mogą być inne niż standardowe Attention.
    """
    linear_types = (torch.nn.Linear,)
    try:
        from bitsandbytes.nn import Linear4bit, Linear8bitLt
        linear_types = (torch.nn.Linear, Linear4bit, Linear8bitLt)
    except ImportError:
        pass

    target = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_types):
            leaf = name.split(".")[-1]
            target.add(leaf)

    # Priorytetyzuj kluczowe moduły attention/feed-forward jeśli istnieją
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

    print("🛠️  Patchowanie konfiguracji modelu (MoE + NVFP4)...")
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if hasattr(config, "quantization_config"):
        print("💡 Usuwanie quantization_config dla natywnej obsługi NVFP4...")
        del config.quantization_config

    print("📥 Ładowanie modelu 120B do VRAM...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        config=config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_impl,  # Fix 1: dynamiczny fallback
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        ignore_mismatched_sizes=True,
    )

    # Fix 2: NIE wywołujemy prepare_model_for_kbit_training — model jest bf16, nie 4-bit
    # Ręcznie mrożemy wagi bazowe
    for param in model.parameters():
        param.requires_grad = False

    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=get_target_modules(model),  # Fix 3: dynamiczne wykrywanie
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
        optim="adamw_torch_fused",   # Fix 2: paged_adamw_8bit jest dla modeli 4-bit
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=8192,
        processing_class=tokenizer,   # Fix 4: `tokenizer` deprecated w TRL 0.29+
        args=training_args,
        formatting_func=formatting_prompts_func,
    )

    print("\n🚀 START: Trening Nemotron-3 Super 120B...")
    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    print(f"\n✅ Trening zakończony! Model zapisany w: {OUTPUT_DIR}")


if __name__ == "__main__":
    train()
