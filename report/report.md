# When Does Cross-Candidate Context Help? A Controlled Study of MP2 Candidate Selection

DASE7506 Advanced Machine Learning Methods · MP2 · Fall 2026  
Student ID: 3036747927

## Abstract

We study whether a generated solution should be scored independently or in the context of three alternatives. The MP2 task supplies four candidate reasoning traces and a fixed 1,024-dimensional embedding for each trace; the system selects one candidate whose **final numeric answer** matches the reference. We compare a training-free answer-vote baseline, an independent neural scorer, three DeepSets variants, and three permutation-equivariant self-attention variants. The neural variants use the same training split, multi-positive listwise objective, and three random seeds. Exact answer-consistency information raises the DeepSets test count from 378 to 385 out of 467; adding embedding-cosine relations lowers it to 380. Attention without explicit relation features also reaches 385/467 (82.44%). DeepSets-answer is the submitted artifact. The paired differences are small, so these results do not establish a general advantage for cross-candidate context.

## 1. Task, data, and research question

Each example contains a question, four candidate reasoning traces, and an embedding array of shape `[4, 1024]`. The official evaluator parses a candidate's final answer after a `####` marker or the last boxed expression, normalizes numerical equivalents, and awards one point when it matches the reference. It does not certify the intermediate reasoning.

The supplied splits contain 4,755 training, 798 validation, and 467 test questions. Every test group contains at least two distinct parseable answers; 355 validation groups meet the same condition. The remaining 443 validation groups are easier: every method in our main experiment selected a correct answer for them. We therefore report both full-validation accuracy and accuracy on the 355 answer-disagreement questions. The latter is a development diagnostic, not a separate data split.

Our question is: **Does information from the other three candidates improve selection, and which relations are useful?** We compare independent scoring with two ways of exchanging candidate information, then compare the same four-candidate architecture with no explicit relation, exact answer equality, or exact equality plus embedding cosine. We expected exact answer agreement to provide useful information on ambiguous groups; the experiments test whether that benefit persists across architectures.

## 2. Methods and training

### 2.1 Shared inputs and decision rule

Each learned scorer receives an L2-normalized 1,024-dimensional whole-trace embedding and seven local numerical features: answer parseability, answer sign, log answer magnitude, integer status, log text length, log numerical-token count, and coverage of numbers appearing in the question. Local-feature means and standard deviations are fitted on **train only** and stored in the checkpoint. The training-free baseline parses all four final answers, selects the most frequent valid answer, and breaks ties by earliest candidate. Neural models average the three seeds' logits and choose the highest-scoring candidate. Inference uses no question IDs, reference answers, or cached choices.

All learned variants use a shared `1031 → 64 → 64` ReLU candidate encoder. The **Independent** scorer processes each candidate without access to the other three. **DeepSets** combines each encoded candidate $h_i$ with the mean and maximum over all four: $[h_i,\operatorname{mean}(h),\operatorname{max}(h),h_i-\operatorname{mean}(h)]$. **Attention** instead uses one unmasked, single-head self-attention layer without position encoding and scores $[h_i,z_i,h_i-z_i,h_i\odot z_i]$. DeepSets and Attention are permutation equivariant: permuting the candidates permutes their scores without giving an index special meaning.

### 2.2 Explicit relation ablations

Within DeepSets, `none` uses only learned four-candidate context; `answer` adds each candidate's exact-answer vote fraction; `answer_cosine` also adds its mean and maximum embedding cosine similarity to the other candidates. Within Attention, the three corresponding modes add no explicit bias, an off-diagonal exact-answer-equality bias, or equality plus off-diagonal embedding-cosine bias to the attention logits. **Attention-none still compares all four candidates** through learned attention. Here, `none` means that no hand-specified relation is supplied. The three variants within each family keep the same architecture and parameter count.

### 2.3 Training protocol

Training references identify every candidate whose parsed final answer is correct. The objective is the negative log of the **sum** of softmax probabilities assigned to all correct candidates. We train on 2,513 informative training groups with fewer than four correct candidates. Every neural configuration uses seeds 17, 23, and 42; AdamW with learning rate $10^{-3}$ and weight decay $10^{-4}$; batch size 128; no dropout; gradient clipping at 1; a 100-epoch maximum; and patience 15. Each seed's checkpoint is selected by its lowest hard-validation loss. The three selected models are ensembled by arithmetic mean of their logits. Test references do not fit weights, feature statistics, or thresholds.

No four-candidate model uses an index feature or positional encoding. We also permuted all four candidates in ten validation questions through all 24 possible orders (240 predictions). After mapping predictions back to the original candidate identities, every selected identity matched the unpermuted prediction. This checks direct position dependence, although generator-style correlations remain possible.

## 3. Eight-method experiment

| Method | Validation /798 | Hard validation /355 | Test /467 | Parameters per seed | Test wall time (s) |
|---|---:|---:|---:|---:|---:|
| Answer majority | 688 (86.22%) | 245 (69.01%) | 323 (69.16%) | 0 | 1.21 |
| Independent | 740 (92.73%) | 297 (83.66%) | 382 (81.80%) | 78,465 | 1.39 |
| DeepSets-none | 737 (92.36%) | 294 (82.82%) | 378 (80.94%) | 78,465 | 1.29 |
| **DeepSets-answer (submitted)** | **739 (92.61%)** | **296 (83.38%)** | **385 (82.44%)** | **78,465** | **1.38** |
| DeepSets-answer-cosine | 739 (92.61%) | 296 (83.38%) | 380 (81.37%) | 78,465 | 1.76 |
| Attention-none | 732 (91.73%) | 289 (81.41%) | 385 (82.44%) | 95,235 | 1.75 |
| Attention-answer | 736 (92.23%) | 293 (82.54%) | 383 (82.01%) | 95,235 | 1.77 |
| Attention-answer-cosine | 737 (92.36%) | 294 (82.82%) | 381 (81.58%) | 95,235 | 1.77 |

Counts come from the official prediction runner and scorer. Wall time is one fresh prediction process plus one independent scoring process, run sequentially with four-thread settings on an Intel Core i7-12700H. The course reference Xeon CPU determines formal eligibility. Parameter counts are **per ensemble member**; the checkpoint contains three members.

The majority baseline struggles on the test bank, where candidate answers disagree by construction. Within DeepSets, explicit answer consistency raises the test count by seven relative to DeepSets-none without adding trainable parameters; cosine similarity does not add a further gain. Within Attention, explicit relation biases raise validation accuracy but do not raise the Attention-none test count. Independent has the highest full and hard-validation counts in this matrix. DeepSets-answer and Attention-none tie for the highest reported test count. A three-question edge over Independent is too small to establish a reliable architectural advantage.

### 3.1 Answer-pattern breakdown for the tied models

The patterns below count multiplicities of distinct, valid parsed final answers. Invalid parses are omitted from the multiplicity and included in “Other.” This breakdown describes the frozen models' results; it is not an additional selection rule.

| Valid-answer pattern | Questions | DeepSets-answer correct | Attention-none correct |
|---|---:|---:|---:|
| 3–1 | 196 | 181 | 179 |
| 2–1–1 | 102 | 74 | 74 |
| 2–2 | 39 | 30 | 30 |
| 1–1–1–1 | 38 | 27 | 29 |
| Other, including invalid parses | 92 | 73 | 73 |

DeepSets-answer has two more correct choices on 3–1 questions; Attention-none has two more when all four valid answers differ. The cells are small, and these patterns should be treated as error-analysis clues. Equal overall counts conceal nine test questions uniquely solved by each model.

## 4. Paired comparison and representative errors

On test, DeepSets-answer corrects nine Independent errors but loses six Independent successes (exact McNemar $p\approx0.607$). Against Attention-none, each model alone gets nine questions right ($p=1.0$). Thus equal or similar total scores conceal different decisions; the paired evidence does not show that one model is reliably stronger. A 95% Wilson interval for 385/467 is approximately 78.73%–85.62%. It describes sampling uncertainty only and does not cover distribution shift or uncertainty from comparing many methods.

In one question, eight shares rise 50% and then fall 25%. DeepSets-answer selects the correct final value, 72; Attention-none selects 168, which adds holdings across years instead of reporting the ending value. On a five-week makeup-cost question with a 10% discount, Attention-none selects 27,000 while DeepSets-answer selects 47,250; the latter effectively charges seven days per week instead of the stated four sessions. In another shared failure, a candidate reaches the correct final answer, 10, through flawed intermediate arithmetic. This illustrates the benchmark's final-answer metric: a numerically correct conclusion receives credit even when its explanation is unsound.

## 5. Efficiency, limitations, and conclusion

The submitted DeepSets-answer checkpoint contains all three seed models and train-fitted feature statistics. It is 952,031 bytes (0.908 MiB), below the 16 MiB inference-asset limit. On the local i7-12700H, fresh test prediction took 1.27 seconds and independent scoring took 0.11 seconds, or 1.38 seconds combined, with four-thread settings and CPU inference. The reference machine determines formal compliance with the 30-second limit. The reproduced environment was Python 3.12.14, NumPy 2.2.6, and PyTorch 2.11.0+cu130. The data integrity checker passed. Per-seed metrics, official predictions and scores, exact commands, and hashes accompany the repository.

Early stopping, rather than the 100-epoch cap, determined the submitted weights. DeepSets-answer seeds 17/23/42 had their lowest hard-validation loss at epochs 8/11/11. Their individual full-validation counts were 740/739/740; logit averaging produced 739. The ensemble was fixed in advance and is not claimed to improve every validation metric. Attention-none selected epochs 5/8/6. Simply raising the epoch cap would not address the observed later validation-loss deterioration.

The evidence is limited to 467 test questions from one supplied generator and one fixed embedding family. The metric checks final numeric answers rather than reasoning validity, and no result establishes generalization to another generator or an audit bank.  Historical hyperparameter-search logs are unavailable in the current workspace, so this report does not assign unverified values to those trials. No test label enters the submitted model or its training procedure.

Our conclusion is narrow: exact answer-consistency information helps this DeepSets scorer, while additional cosine relations and a more expressive attention layer do not consistently improve the result. DeepSets-answer is the reproducible submitted configuration at **385/467 = 82.44111349036403%**; Attention-none reaches the same count at a higher parameter and local inference cost. The paired differences do not establish a decisive advantage for either architecture.

## Attribution and AI disclosure

Questions and references derive from GSM8K (OpenAI, 2021), under its retained MIT license. The course prepared candidate traces and fixed representations with Qwen3-0.6B, revision `c1899de289a04d12100db370d81485cdf75e47ca`; this submission downloads or distributes no Qwen weights. We use the supplied parser and evaluator without modification. Substantive Codex assistance was used for model and ablation implementation, experiment execution and analysis, and drafting this report; the student is responsible for checking the implementation and submitted claims. See the course guide, data manifest, license, and notice files in the repository.
