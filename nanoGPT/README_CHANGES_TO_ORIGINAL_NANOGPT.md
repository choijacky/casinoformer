
# Changes made to nanoGPT for our Deep Learning Project

Despite some device name changes (mainly from "cuda" to "mps"), the code was mostly left intact. In the following, we document substantial changes made to the original nanoGPT repo authored by Andrej Karpathy.

1. We added the "inference_bench.py" file, which is a copy of the original `sample.py`, extended by a loop to compute the perplexity. 
2. To compute the perplexity, we also added the function `generate_with_probs` in the `model.py` file, which, other than the original `generate`, outputs only the generated token, and its logarithmic conditional probability dependent on the context. We extended model.py with cache capabilities, which is vital for bringing down O(L^2) complexity to O(L) in autoregressive generation.
3. We added the wikitext-2 dataset in the `data` folder including a `prepare.py` script based on the `openwebtext/prepare.py` version.
5. We added the folder `new_out` if new runs should be generatded. Our runs are stored in `out`.
