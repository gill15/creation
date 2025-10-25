# creation

Train a tiny language model on the `book_cleaned.txt` corpus from Hugging Face. The
project provides a lightweight fine-tuning script that downloads the corpus and
runs a short training session on a compact GPT-2 style model so experimentation
remains CPU friendly.

## Requirements

Install the Python dependencies before running the training script:

```bash
python -m venv .venv           # optional but recommended
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

1. **Download and cache the dataset** – The script pulls `book_cleaned.txt`
   from the Hugging Face Hub the first time it runs. No manual download is
   required, but you need internet access for that initial execution. The file
   is cached under `~/.cache/huggingface` so later runs reuse it.
2. **Launch training** with the defaults:

   ```bash
   python scripts/train_tiny_model.py
   ```

   This spins up a short fine-tuning run using the `distilgpt2` checkpoint and
   writes the resulting weights and tokenizer to the `tiny-book-model`
   directory.

3. **Customize the run** (optional) by passing flags:

   ```bash
   python scripts/train_tiny_model.py \
       --max-chars 1000000 \
       --block-size 256 \
       --batch-size 4 \
       --epochs 3 \
       --output-dir ./experiments/distilgpt2-run
   ```

   Key options:

   * `--repo-id`: Hugging Face dataset repository that hosts `book_cleaned.txt`.
   * `--filename`: File to download from the repository.
   * `--max-chars`: Maximum number of characters to read from the corpus (0 for the full file).
   * `--model`: Base causal language model checkpoint to fine-tune.
   * `--output-dir`: Where the trained model and tokenizer are stored.

If you plan to use a private dataset or model, authenticate with Hugging Face
first (`huggingface-cli login`). Public resources like `bookcorpusopen` do not
require authentication.
