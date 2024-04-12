from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer
# import torch

max_seq_length = 2048
dtype = None
load_in_4bit = True


def load_model():
    print("Loading model")
    return "", ""
    # model, tokenizer = FastLanguageModel.from_pretrained(
    #     model_name="scb10x/typhoon-7b",
    #     max_seq_length=max_seq_length,
    #     dtype=dtype,
    #     load_in_4bit=load_in_4bit,
    # )

    # model = FastLanguageModel.get_peft_model(
    #     model,
    #     r=16,
    #     target_modules=[
    #         "q_proj",
    #         "k_proj",
    #         "v_proj",
    #         "o_proj",
    #         "gate_proj",
    #         "up_proj",
    #         "down_proj",
    #     ],
    #     lora_alpha=16,
    #     lora_dropout=0,
    #     bias="none",
    #     use_gradient_checkpointing=True,
    #     random_state=3407,
    #     use_rslora=False,
    #     loftq_config=None,
    # )
    # return model, tokenizer


# alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

# ### Instruction:
# {}

# ### Input:
# {}

# ### Response:
# {}"""

# EOS_TOKEN = tokenizer.eos_token


# def formatting_prompts_func(examples):
#     instructions = examples["instruction"]
#     inputs = examples["input"]
#     outputs = examples["output"]
#     texts = []
#     for instruction, input, output in zip(instructions, inputs, outputs):
#         text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
#         texts.append(text)
#     return {
#         "text": texts,
#     }


# pass

# dataset = load_dataset("./typhoon", split="train")
# dataset = dataset.map(
#     formatting_prompts_func,
#     batched=True,
# )

# trainer = SFTTrainer(
#     model=model,
#     tokenizer=tokenizer,
#     train_dataset=dataset,
#     dataset_text_field="text",
#     max_seq_length=max_seq_length,
#     dataset_num_proc=2,
#     packing=False,
#     args=TrainingArguments(
#         per_device_train_batch_size=2,
#         gradient_accumulation_steps=4,
#         warmup_steps=5,
#         max_steps=60,
#         learning_rate=2e-4,
#         fp16=not torch.cuda.is_bf16_supported(),
#         bf16=torch.cuda.is_bf16_supported(),
#         logging_steps=1,
#         optim="adamw_8bit",
#         weight_decay=0.01,
#         lr_scheduler_type="linear",
#         seed=3407,
#         output_dir="outputs",
#     ),
# )

# # @title Show current memory stats
# gpu_stats = torch.cuda.get_device_properties(0)
# start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
# max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
# print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
# print(f"{start_gpu_memory} GB of memory reserved.")

# trainer_stats = trainer.train()

# # @title Show final memory and time stats
# used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
# used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
# used_percentage = round(used_memory / max_memory * 100, 3)
# lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
# print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
# print(
#     f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
# )
# print(f"Peak reserved memory = {used_memory} GB.")
# print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
# print(f"Peak reserved memory % of max memory = {used_percentage} %.")
# print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


# FastLanguageModel.for_inference(model)
# inputs = tokenizer(
#     [
#         alpaca_prompt.format(
#             "ขอมาตรฐานการเรียนรู้",  # instruction
#             "หน่วยการเรียนรู้ที่ 1  วิชาภาษาไทย ระดับชั้นประถมศึกษาปีที่ 1 ฉันรักเธอ",  # input
#             "",  # output - leave this blank for generation!
#         )
#     ],
#     return_tensors="pt",
# ).to("cuda")

# outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
# tokenizer.batch_decode(outputs)

# # alpaca_prompt = Copied from above
# FastLanguageModel.for_inference(model)
# inputs = tokenizer(
#     [
#         alpaca_prompt.format(
#             "ทำลำดับฟีโบนาชีต่อ",  # instruction
#             "1, 1, 2, 3, 5, 8",  # input
#             "",  # output - leave this blank for generation!
#         )
#     ],
#     return_tensors="pt",
# ).to("cuda")

# text_streamer = TextStreamer(tokenizer)
# _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)


# model.save_pretrained("thai_typhoon_model")  # Local saving
# # model.push_to_hub("your_name/lora_model", token = "...") # Online saving

# if False:
#     from unsloth import FastLanguageModel

#     model, tokenizer = FastLanguageModel.from_pretrained(
#         model_name="thai_typhoon_model",  # YOUR MODEL YOU USED FOR TRAINING
#         max_seq_length=max_seq_length,
#         dtype=dtype,
#         load_in_4bit=load_in_4bit,
#     )
#     FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

# # alpaca_prompt = You MUST copy from above!

# inputs = tokenizer(
#     [
#         alpaca_prompt.format(
#             "ทำลำดับฟีโบนาชีต่อ",  # instruction
#             "",  # input
#             "",  # output - leave this blank for generation!
#         )
#     ],
#     return_tensors="pt",
# ).to("cuda")

# outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
# tokenizer.batch_decode(outputs)

# if False:
#     # I highly do NOT suggest - use Unsloth if possible
#     from peft import AutoPeftModelForCausalLM
#     from transformers import AutoTokenizer

#     model = AutoPeftModelForCausalLM.from_pretrained(
#         "thai_typhoon_model",  # YOUR MODEL YOU USED FOR TRAINING
#         load_in_4bit=load_in_4bit,
#     )
#     tokenizer = AutoTokenizer.from_pretrained("thai_typhoon_model")

# # model.save_pretrained_merged("thai_typhoon_model", tokenizer, save_method = "merged_16bit",)
# model.push_to_hub_merged(
#     "hf/airbornharsh/thai_typhoon_model",
#     tokenizer,
#     save_method="merged_16bit",
#     token="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",
# )

# # Merge to 4bit
# if False:
#     model.save_pretrained_merged(
#         "model",
#         tokenizer,
#         save_method="merged_4bit",
#     )
# if False:
#     model.push_to_hub_merged("hf/model", tokenizer, save_method="merged_4bit", token="")

# # Just LoRA adapters
# if False:
#     model.save_pretrained_merged(
#         "model",
#         tokenizer,
#         save_method="lora",
#     )
# if False:
#     model.push_to_hub_merged("hf/model", tokenizer, save_method="lora", token="")

# # Save to 8bit Q8_0
# if False:
#     model.save_pretrained_gguf(
#         "model",
#         tokenizer,
#     )
# if False:
#     model.push_to_hub_gguf("hf/model", tokenizer, token="")

# # Save to 16bit GGUF
# if False:
#     model.save_pretrained_gguf("model", tokenizer, quantization_method="f16")
# if False:
#     model.push_to_hub_gguf("hf/model", tokenizer, quantization_method="f16", token="")

# # Save to q4_k_m GGUF
# if False:
#     model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")
# if False:
#     model.push_to_hub_gguf(
#         "hf/model", tokenizer, quantization_method="q4_k_m", token=""
#     )
