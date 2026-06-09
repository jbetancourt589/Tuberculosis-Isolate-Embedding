import os
import json
import torch
import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM

MODEL_NAME = "InstaDeepAI/nucleotide-transformer-500m-human-ref"

FASTA_DIR = r"Data/IR_Variable"
TARGETS_JSON = r"Data/cryptic_targets_all.json"

OUTPUT_DIR = "outputs"
EMBEDDINGS_OUT = os.path.join(OUTPUT_DIR, "isolate_embeddings.npy")
METADATA_OUT = os.path.join(OUTPUT_DIR, "isolate_metadata.csv")

MAX_LENGTH = 1000

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

print("Loading model...")
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, trust_remote_code=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

print("Device:", device)

with open(TARGETS_JSON, "r") as f:
    targets = json.load(f)

def clean_sequence(seq):
    seq = str(seq).upper()
    return "".join(base for base in seq if base in "ACGTN")

@torch.no_grad()
def embed_sequence(sequence):
    sequence = clean_sequence(sequence)

    if len(sequence) < 50:
        return None

    sequence = sequence[:MAX_LENGTH]

    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    outputs = model(**inputs, output_hidden_states=True)

    last_hidden_state = outputs.hidden_states[-1]
    attention_mask = inputs["attention_mask"].unsqueeze(-1)

    embedding = (last_hidden_state * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

    return embedding.squeeze().cpu().numpy()

def embed_isolate(fasta_path):
    unit_embeddings = []

    records = list(SeqIO.parse(fasta_path, "fasta"))

    i = 0
    while i < len(records) - 2:
        before_record = records[i]
        gene_record = records[i + 1]
        after_record = records[i + 2]

        before_header = before_record.description
        after_header = after_record.description

        if "BEFORE" in before_header and "AFTER" in after_header:
            combined_sequence = (
                str(before_record.seq).upper()
                + str(gene_record.seq).upper()
                + str(after_record.seq).upper()
            )

            emb = embed_sequence(combined_sequence)

            if emb is not None:
                unit_embeddings.append(emb)

            i += 3
        else:
            i += 1

    if len(unit_embeddings) == 0:
        return None, 0

    isolate_embedding = np.mean(unit_embeddings, axis=0)

    return isolate_embedding, len(unit_embeddings)
fasta_files = [
    f for f in os.listdir(FASTA_DIR)
    if f.endswith(".fasta")
]
#only works with the first 5 fasta files
fasta_files = fasta_files[:5]

print("FASTA files found:", len(fasta_files))

all_embeddings = []
metadata = []

for filename in tqdm(fasta_files):
    isolate_id = filename.replace("_IR_Genes.fasta", "")
    fasta_path = os.path.join(FASTA_DIR, filename)

    if isolate_id not in targets:
        print(f"Skipping {isolate_id}: no target labels found")
        continue

    isolate_embedding, gene_count = embed_isolate(fasta_path)

    if isolate_embedding is None:
        print(f"Skipping {isolate_id}: no valid gene embeddings")
        continue

    all_embeddings.append(isolate_embedding)

    row = {
        "isolate_id": isolate_id,
        "filename": filename,
        "num_gene_units_embedded": gene_count
    }

    row.update(targets[isolate_id])
    metadata.append(row)

embeddings_array = np.vstack(all_embeddings)
metadata_df = pd.DataFrame(metadata)

np.save(EMBEDDINGS_OUT, embeddings_array)
metadata_df.to_csv(METADATA_OUT, index=False)

print("Done.")
print("Embeddings saved to:", EMBEDDINGS_OUT)
print("Metadata saved to:", METADATA_OUT)
print("Embeddings shape:", embeddings_array.shape)
print("Metadata shape:", metadata_df.shape)