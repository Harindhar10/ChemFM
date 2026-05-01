from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
import torch
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16
                )

                
model = AutoModelForCausalLM.from_pretrained('harindhar10/OLMo-7B-fsdp-Pubchem-500k-1epochs-eos',
                                            trust_remote_code=True,
                                            low_cpu_mem_usage = True,
                                            quantization_config= bnb_config,
                                            dtype=torch.float16)

task_type = "CAUSAL_LM"

model = prepare_model_for_kbit_training(
    model, use_gradient_checkpointing=True
)

lora_cfg = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=task_type,
)
model = get_peft_model(model, lora_cfg)

model.eval()
model.to('cuda')

model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
model.generation_config.cache_implementation = 'static'