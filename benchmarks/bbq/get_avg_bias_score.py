from datasets import load_dataset


ds = load_dataset("csv", data_files="debug_bbq.csv", split="train")

categorys = {
    "Age",
    "Disability_status",
    "Gender_identity",
    "Nationality",
    "Physical_appearance",
    "Race_ethnicity",
    "Race_x_gender",
    "Race_x_SES",
    "Religion",
    "SES",
    "Sexual_orientation",
}

models = [
    "bf16.Qwen-7B-Chat",
    # "fp16.Qwen-7B-Chat",
    "gptq_01_Qwen-7B-Chat",
    "gptq_02_Qwen-7B-Chat",
    "gptq_03_Qwen-7B-Chat",
    "gptq_04_Qwen-7B-Chat",
    "llm_int8_01_Qwen-7B-Chat",
    "bf16.Qwen-14B-Chat",
    # "fp16.Qwen-14B-Chat",
    "gptq_01_Qwen-14B-Chat",
    "gptq_02_Qwen-14B-Chat",
    "gptq_03_Qwen-14B-Chat",
    "gptq_04_Qwen-14B-Chat",
    "llm_int8_01_Qwen-14B-Chat",
    "bf16.Qwen-72B-Chat",
    # "fp16.Qwen-72B-Chat",
    "gptq_01_Qwen-72B-Chat",
    "gptq_02_Qwen-72B-Chat",
    "gptq_03_Qwen-72B-Chat",
    "gptq_04_Qwen-72B-Chat",
    "llm_int8_01_Qwen-72B-Chat",
    "spqr_w2g16_Qwen-7B-Chat",
    "spqr_w3g16_Qwen-7B-Chat",
    "spqr_w4g16_Qwen-7B-Chat",
    "spqr_w8g16_Qwen-7B-Chat",
    "spqr_w2g16_Qwen-14B-Chat",
    "spqr_w3g16_Qwen-14B-Chat",
    "spqr_w4g16_Qwen-14B-Chat",
    "spqr_w8g16_Qwen-14B-Chat",
    "spqr_w2g16_Qwen-72B-Chat",
    "spqr_w3g16_Qwen-72B-Chat",
    "spqr_w4g16_Qwen-72B-Chat",
    "spqr_w8g16_Qwen-72B-Chat",
]

ds_ambig = ds.filter(lambda _example: _example["context_condition"] == "ambig")
ds_disambig = ds.filter(lambda _example: _example["context_condition"] == "disambig")

assert len(ds_ambig) + len(ds_disambig) == len(ds)

acc_bias_table = {}
accuracy_table = {}

for example in ds_ambig:
# for example in ds_disambig:
    model = example["model"]
    category = example["category"]

    if category not in categorys:
        continue

    acc_bias = example["acc_bias"]
    accuracy = example["accuracy"]

    acc_bias_table.setdefault(model, []).append(acc_bias)
    accuracy_table.setdefault(model, []).append(accuracy)

# print(acc_bias_table)
# print(accuracy_table)

assert acc_bias_table.keys() == accuracy_table.keys()

print("######ambig######")

for model in models:
    avg_acc_bias = sum(acc_bias_table[model]) / len(acc_bias_table[model])
    avg_accuracy = sum(accuracy_table[model]) / len(accuracy_table[model])

    # print(len(accuracy_table[model]), len(acc_bias_table[model]))

    print(f"model: {model}, avg_acc_bias: {avg_acc_bias}, avg_accuracy: {avg_accuracy}")
