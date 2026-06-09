# LLMTB

LLMTB is an early-stage tuberculosis genomics project for creating isolate-level embeddings from FASTA sequences. The long-term goal is to use these embeddings in a vector database so a new tuberculosis isolate can be compared against past isolates with similar genomes.

Each isolate is paired with antibiotic susceptibility labels. Once the embeddings are indexed, a nearest-neighbor search could support exploratory resistance prediction: for example, if the five closest historical isolates are resistant to a drug, the new isolate may be more likely to behave similarly. This repository currently focuses on generating embeddings and metadata; vector database indexing and nearest-neighbor querying are planned future work.

## Project Status

This project currently:

- Loads tuberculosis isolate FASTA files from `Data/IR_Variable`
- Loads antibiotic susceptibility labels from `Data/cryptic_targets_all.json`
- Embeds each gene unit, defined as `BEFORE region + gene + AFTER region`, using `InstaDeepAI/nucleotide-transformer-500m-human-ref`
- Averages gene-unit embeddings into one fixed-length embedding per isolate
- Saves embeddings as a NumPy array and isolate metadata as a CSV file

The vector database workflow is not implemented yet.

## Pipeline

```text
FASTA isolate
      |
      v
BEFORE + gene + AFTER
      |
      v
Nucleotide Transformer
      |
      v
Gene-unit embeddings
      |
      v
Average
      |
      v
One isolate embedding
      |
      v
isolate_embeddings.npy
      |
      v
Future vector database (FAISS)
      |
      v
Nearest-neighbor search
      |
      v
Resistance prediction support
```

## Repository Structure

```text
.
+-- Data/
|   +-- IR_Variable/              # Input FASTA files for isolates
|   +-- cryptic_targets_all.json  # Antibiotic susceptibility labels by isolate
+-- embed_isolates.py             # Main embedding pipeline
+-- test.py                       # Smoke test for model loading and one FASTA sequence
+-- README.md
```

Generated files are written by the script to:

```text
outputs/
+-- isolate_embeddings.npy
+-- isolate_metadata.csv
```

## Requirements

This project uses Python and the Hugging Face Transformers ecosystem.

Install the required packages:

```bash
pip install torch numpy pandas biopython tqdm transformers
```

For GPU acceleration, install the PyTorch build that matches your CUDA version from the official PyTorch installation instructions.

## Embedding Model

This project currently uses the pretrained model:

```text
InstaDeepAI/nucleotide-transformer-500m-human-ref
```

No additional fine-tuning is currently performed. The model is used as a feature extractor to generate fixed-length genomic embeddings.

## Data Expectations

The embedding script expects:

- FASTA files in `Data/IR_Variable`
- FASTA filenames ending in `.fasta`
- Isolate IDs derived from filenames by removing `_IR_Genes.fasta`
- A target-label JSON file at `Data/cryptic_targets_all.json`
- Matching isolate IDs between the FASTA filenames and the target-label JSON

The FASTA parsing logic looks for groups of three records:

```text
BEFORE region
gene region
AFTER region
```

When a valid group is found, the script concatenates the BEFORE, gene, and AFTER sequences, embeds the combined sequence, and later averages all valid gene-region embeddings for that isolate.

The BEFORE and AFTER regions are included because regulatory mutations outside coding regions may contribute to antibiotic resistance and gene expression changes.

Each isolate contains many gene units. The embedding generated for each gene unit is averaged to produce a single fixed-length embedding representing the overall genomic characteristics of the isolate.

## Usage

Run a quick smoke test:

```bash
python test.py
```

Generate isolate embeddings:

```bash
python embed_isolates.py
```

The current script is configured to process only the first five FASTA files:

```python
fasta_files = fasta_files[:5]
```

Remove or change that line in `embed_isolates.py` to process more isolates.

## Outputs

`isolate_embeddings.npy` contains a 2D NumPy array:

```text
number_of_isolates x embedding_dimension
```

Each row represents one isolate embedding, and each column corresponds to one embedding dimension produced by the pretrained nucleotide transformer.

`isolate_metadata.csv` contains:

- `isolate_id`
- `filename`
- `num_gene_units_embedded`
- Antibiotic susceptibility labels from `cryptic_targets_all.json`

The CSV provides the mapping between each embedding and its corresponding isolate, along with antibiotic susceptibility labels and embedding statistics. It acts as the lookup table for interpreting the rows in `isolate_embeddings.npy`.

These two files are meant to stay aligned by row index. Row `i` in `isolate_embeddings.npy` corresponds to row `i` in `isolate_metadata.csv`.

## Intended Future Workflow

The planned next step is to load `isolate_embeddings.npy` into a vector database, keep `isolate_metadata.csv` attached as metadata, and query the database with embeddings generated from new patient isolates.

A future workflow may look like:

1. Generate an embedding for a new tuberculosis isolate.
2. Search the vector database for the closest historical isolate embeddings.
3. Retrieve antibiotic susceptibility labels for the nearest isolates.
4. Use the labels from the nearest isolates as supporting evidence for resistance or susceptibility patterns.

This should be treated as a research and decision-support workflow, not a standalone clinical diagnostic system.

## Privacy and Data Notes

Before uploading this project to GitHub, confirm that the included data files do not contain protected health information or any data that should remain private. If the FASTA files, labels, or generated outputs are sensitive, exclude them from the public repository and provide instructions for placing the data locally.

## License

No license has been selected yet. Add a license before publishing if you want others to know how they may use, modify, or distribute this code.
