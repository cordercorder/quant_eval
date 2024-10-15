from datasets import load_dataset, load_from_disk

# c4_en = load_dataset("allenai/c4", "en", split='validation', data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"})
# c4_en.save_to_disk("/mnt/d/datasets/ppl/c4/en.c4-validation.00000-of-00008.json.gz")


# c4_en = load_dataset("allenai/c4", "en", split='validation')
# c4_en.save_to_disk("/mnt/d/datasets/ppl/c4/en.c4-validation")

# ptb ok
# ptb_text_only = load_dataset("ptb_text_only", "penn_treebank", split="test")
# ptb_text_only.save_to_disk("/mnt/d/datasets/ppl/ptb/ptb_test.bin")


# print('stream allenai/c4')
# ds = load_dataset("allenai/c4", "en", split='validation', data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"}, streaming=True)
# for item in ds:
#     print(item)
#     break


# print("c4")

# c4_en = load_dataset("c4", "en", split='validation')
# c4_en.save_to_disk("/mnt/d/datasets/ppl/c4/en.c4-validation")

# print("./c4.py")

# c4_en = load_dataset("./c4.py", "en", split='validation')
# c4_en.save_to_disk("/mnt/d/datasets/ppl/c4/en.c4-validation.00000-of-00008")
# c4_en.save_to_disk("/mnt/d/datasets/ppl/c4/en.c4-validation")


wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
wikitext.save_to_disk("/mnt/d/datasets/ppl/wikitext/wikitext2_test")