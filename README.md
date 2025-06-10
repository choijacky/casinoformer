# High Attention Scores and Where to Find Them

The detailed project report can be found at [here](./high_attention_scores_and_where_to_find_them.pdf). We received the following two reviews from the teaching assistants:

---

### Official Review of Submission10 by Reviewer aRsD
Official ReviewReviewer aRsD15 Feb 2024 at 14:32 (modified: 21 Feb 2024 at 11:43)Program Chairs, Reviewers Submitted, Reviewer aRsD, AuthorsRevisions
Reviewing Instructions: I have read the reviewing instructions in moodle., The project has a proposal and I have read it.
### Strengths:
This is an excellent piece of work, studying an interesting and well motivated question of whether one can reduce the expense of attention computation by identifying high attention scores. The paper is mostly well written, and the related work and experimental setup are clear. The proposed methods presents mixed results (as often happens in research), but the authors do a great job of analysing their results and understanding why some of their methods (e.g. using FAVOR+) failed. Overall, a very nice and impressive work for the DL project, congrats!

### Weaknesses:
The results are mixed, but this is not important for the project, though would be if you wanted to submit to a conference.
Some of the notation/descriptions regarding the tree structure e.g. Figure 1, are quite difficult to parse.
It's not clear if the reduced attention score computations will actually lead to efficiency gains, due to the tree-structure vs parallelisability of attention.

Rating Quality Of Paper: 6.0: Perfect score

Rating Execution Of Idea: 6.0: Perfect score

Rating Creativity: 6.0: Perfect score

---

### Official Review of Submission10 by Reviewer YRVU
Official ReviewReviewer YRVU15 Feb 2024 at 11:28 (modified: 21 Feb 2024 at 11:43)Program Chairs, Reviewers Submitted, Reviewer YRVU, AuthorsRevisions
Reviewing Instructions: I have read the reviewing instructions in moodle., The project has a proposal and I have read it.
Strengths:
This paper proposes a partial tree-traversal algorithm to approximate attention scores in linear transformers, where the softmax is replaced with a kernelized version. The paper is dense in reference methods and provides good justifications for the methods used and the theoretical aspects.

### Quality of paper:
I really liked the title and some paragraph names are witty, it for sure catches attention.
The deviations from the proposal are justified. The authors generally addressed the proposal feedback, e.g. making Fig. 1 more explanatory, albeit I think it could still benefit from a longer explanation.
Well-written, clear and focused language. The paper is dense in theory and the authors mostly do a good job in terms of providing mathematical details and explaining their ideas. The literature review is concise and provides details about the baseline methods.
### Execution of idea
Clean code. Very nice, mostly informative README! I found the demonstrations for “Alignment” and “FAVOR+” particularly useful.
The authors include several different methods from the literature to estimate probability mass.
The authors elaborate well on the variance of the FAVOR+ algorithm
### Creativity of solution
The project combines relevant ideas from the literature with a sufficient amount of complexity both in theory and implementation.
### Weaknesses:
Quality of paper:
There is no abstract.
As noted earlier, the paper is very dense in information which sometimes breaks the readers’ attention. I believe some methods or mathematical details could have been expanded in the appendix.
The literature review is mostly limited to the baseline methods that the authors use. Since the authors use a diverse set of methods, I don’t find it particularly problematics but a broader view on the literature in the intro could have placed the paper better.
Execution of idea
Due to the points raised above, the exact method proposed in the paper is not very clear in the beginning. I was able to understand it better after going through the code. This was also raised in the proposal feedback and not clearly communicated in the report. For example, you load a pretrained checkpoint from Hugging Face to your GPT model (https://gitlab.ethz.ch/devirac/casinoformer-deep-learning-project-hs23/-/blame/main/nanoGPT/inference_bench.py#L66), and then replace the attention method by your approximate kernel attention. I think some of intricacies, which may help you explain the high perplexity scores in Fig. 2, of your methodology is missing in the report.
Since you are relying on sampling methods, I believe the analysis on the standard deviation is a bit lacking in the paper. You could have evaluated Fig.2 on more seeds (more than two) given that you use a pre-trained model.
I think a standard Linear Attention baseline could have improved the paper but I am aware that this could have been hard given the time constraints and the authors did a good job in terms of curating a diverse baseline for their method.
One minor thing that could enhanced the paper could be a small runtime analysis of the implementation.

Rating Quality Of Paper: 5.75: Excellent

Rating Execution Of Idea: 5.75: Excellent

Rating Creativity: 6.0: Perfect score

---

The remainder of this README acts to ensure reproducibility of our results and 

## Project description
This repo contains the code to run the algorithm proposed for our Deep Learning project. It takes the [nanoGPT](https://github.com/karpathy/nanoGPT/tree/master) repo as the base structure and we have added our own changes to accompany our attention implementation. (For more details, see [here](nanoGPT/README_CHANGES_TO_ORIGINAL_NANOGPT.md))

`np_attention.py` contains the main script that wraps our algorithm in the `attention_kernel` class. We have implemented 6 methods to obtain the probability mass according to which we descend the tree. The method can be selected using the `self.method` attribute when initialising the `attention_kernel` class. To get perplexity score of our model variants, we have implemented the evaluation in `inference_bench.py`. In addition, we have  a `kernel.py` file, where we have tested the approximation errors of various kernel implementations.

## Install
```
pip install -r requirements.txt
```
If a line-by-line runtime profiling is desired, additionally install the `line-profiler` package.
```
pip install line-profiler
```

## Perplexity
To obtain the perplexity scores, run
```
python nanoGPT/inference_bench.py [OPTIONS]
``` 
### Options
These options are all positional arguments in this specified order:
1. `attention_method`: Chooses the attention variant to be used for perplexity evaluation. One of `tree-less`, `alignment`, `uniform`, `time-decay`, `FAVOR+`, `FAVOR+ReLU`.
2. `number_of_terms_exponent`: A value in {0.2, 0.3, ... , 0.9} which describes the number of buds expanded. Irrelevant for `tree-less`.
3.  `percentage_of_test_set`: What ratio of the dataset should be used. For the runs, a value of 0.2 was used.
4.  `offset`: What offset should be used in the dataset. E.g. for `offset = 0.2`, it skips the first 20% of the dataset. For the runs, a multiple of `percentage_of_test_set` was used.

Our results can be found in the `/results` folder. New test-runs are added to the `out` folder by default. To repeat runs, one can read out the used parameters given in the name of a file, e.g. the file name "nanoGPT/out/perplexity_uniform_0.2_fast_sampling_0.2_0.4.txt" implies that `uniform` was used as the `attention_method`, with `number_of_terms_exponent = 0.2`, `percentage_of_test_set = 0.2` and `offset = 0.4`.

Note that a file consists of multiple runs, each with another offset given in the description of the corresponding run, and only final perplexity score is relevant, as the previous values are accumulated.

**Example**: For a run with `attention_method = uniform`, `number_of_terms_exponent = 0.9`, `percentage_of_test_set = 0.2` and `offset = 0.4`
```
python nanoGPT/inference_bench.py uniform 0.9 0.2 0.4
```

## Line profiling
We have decorated the main `forward` loop of our `attention_kernel` class with `@profiler`. This will return a line-by-line profiling of each line of code to provide additional information about the costs of each operation.

The first time, the user is required to run
```
kernprof -lv np_attention.py
```
afterwards, the profiling will be done automatically, when running `np_attention.py` as a main script.

## Test example
To ensure that our algorithm acts as intended, we have set up a verbose version of our algorithm that will print the actions of our sampling. It can be activated by setting `verbose = True`. In addition, we have set up a test example with which we can easily verify correctness. Our test example sets `L = 8`, `D = 64`, `P = 1`, `H = 1` and `E = 1.0`, since this will make the outputs easy to follow. In addition, we generate random keys, queries and values, but set on query equal to one of the keys (in our case, the key at `l=3`). This will ensure that one attention value is high and our alogrithm should be able to descend to that leaf if our implementation is done correctly. Some methods are highlighted specifically.

### Alignment method

By setting `method = alignment`, we can see that the algorithm correctly expands node 11 as anticipated. We also observe that alignments are split correctly and that the big alignment value corresponds to a high sampling probability.
```
Starting at start index:  0
We currently have  1  non leaf buds and  1  number of buds

start sampling
Alignments:  [[119.905523]]
We have pms:  [1.] and we have sampled  [0]
that correspond to indices  [0]
We are expanding  [0]  into  [1]  and  [2]
new buds_indices:  [1 2]

start sampling
Alignments:  [[99.26277615 20.64274685]]
We have pms:  [0.90394498 0.09605502] and we have sampled  [0]
that correspond to indices  [1]
We are expanding  [1]  into  [3]  and  [4]
new buds_indices:  [3 2 4]

start sampling
Alignments:  [[11.83204469 20.64274685 87.43073145]]
We have pms:  [0.00870705 0.00982318 0.98146977] and we have sampled  [2]
that correspond to indices  [4]
We are expanding  [4]  into  [9]  and  [10]
new buds_indices:  [ 3  2  9 10]
```
### FAVOR+
We have mentioned in **Section 4.4** of the report that the variance of the approximation renders this method useless. As such, evaluating **Equation (9)** with our test example parameters, we yield a standard deviation to mean ratio of ~264. Since our alignment approximation cannot be negative, we produce a lot of low values to offset the high standard deviation to get a lower mean value. This can be seen in the `Phi Alignments`, which are all near zero values.
```
Starting at start index:  0
We currently have  1  non leaf buds and  1  number of buds

start sampling
Phi Alignments:  [[0.29802295]]
Alignments:  [[119.905523]]
We have pms:  [1.] and we have sampled  [0]
that correspond to indices  [0]
We are expanding  [0]  into  [1]  and  [2]
new buds_indices:  [1 2]

start sampling
Phi Alignments:  [[0.26115119 0.03687176]]
Alignments:  [[99.26277615 20.64274685]]
We have pms:  [0.841572 0.158428] and we have sampled  [0]
that correspond to indices  [1]
We are expanding  [1]  into  [3]  and  [4]
new buds_indices:  [3 2 4]

start sampling
Phi Alignments:  [[0.16345653 0.03687176 0.09769466]]
Alignments:  [[11.83204469 20.64274685 87.43073145]]
We have pms:  [0.57206165 0.08602867 0.34190968] and we have sampled  [0]
that correspond to indices  [3]
We are expanding  [3]  into  [7]  and  [8]
new buds_indices:  [7 2 4 8]
```
## Kernel Analysis
The `kernel.py` script enables detailed analysis of various kernel behaviors. It provides insights into mean, standard deviation, and their relationships across diverse settings and interpolation levels and visualises them.

To execute the script the following command can be run:

```
python kernel.py [OPTIONS]
```
### Available Options

- `--method`: Specifies the method for kernel analysis. One of the following methods can be chosen: 
    - `"FAVOR+"`
    - `"FAVOR+ReLU"`
    - `"positive_alignment"`
    - `"RandomFourierFeatures"`
- `--D`: Sets the number of dimensions to N. Default is 64.
- `--R`: Definse the number of output dimensions. For most methods, the default is 128; however, for "RandomFourierFeatures", it defaults to 64.
- `--orth`: Include this flag to control orthogonal features under specific conditions based on the method and dimensions.
- `--query_norm_value`: Sets the norm value for the query to N.
- `--key_norm_value`: Sets the norm value for the key to N.
- `--scaling`: Additional scaling factor applied to adjust variance/discriminative behavior.
- `--exp`: Specifies the exponent value used for certain methods. Higher values increase discriminative power at the cost of higher variance.

**Example**

`python kernel.py --method "FAVOR+" --D 64 --R 128 --orth --query_norm_value 8 --key_norm_value 8 --scaling 1.0 --exp 1.0`

## Graphics
While most graphics are generated on [diagrams](https://app.diagrams.net), some of the graphics are generated using `graphics/GPT2.ipynb`, which we have run on Google Colab.

![](/graphics/softmax_distribution5.png)
*Softmax activations of GPT-2 model in layer 5 for all heads*
