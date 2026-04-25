# Training + Data Pipeline Owner

You own everything from **raw data → ready batch → training step**.
You are responsible for building the pipeline that makes that logic possible later.

## What exactly you should work on

### 1. Data curation

They should handle:
● collecting all training sources
● cleaning bad samples
● deduplication
● removing corrupted / low-quality examples
● normalizing format
● adding metadata like source, domain, difficulty, bucket, teacher preference if needed
● train / val / test split
This is important because your retrieval side later also stores metadata like embedding, bucket
id, teacher id, teacher soft targets, and other fields, so the whole system benefits if the dataset
is already structured cleanly.

### 2. Data pipeline for training

They should build:
● final dataset schema
● preprocessing functions
● tokenization


● packing / truncation / padding
● dataloader
● batching strategy
● shuffling / sampling
● optional bucketed batching by length or domain
● teacher-cache loading if you precompute teacher outputs
This is the part you said you need: **how to build the data pipeline for training**.
So this person should own the full path from raw records to model-ready tensors.

### 3. Teacher-output preparation

Because this is distillation, you should also manage:
● teacher forward pass hookup, or
● precomputed teacher logits / soft targets cache
● mapping each sample to the selected teacher or teacher outputs
So you should make sure every sample can cleanly produce:
● input ids
● labels
● teacher target/logits
● metadata

### 4. Baseline training loop

you should implement:


● student forward pass
● teacher forward pass or teacher-cache read
● KD loss
● CE loss
● regularization terms
● backprop
● optimizer step
● scheduler
● logging (wandb or anything similar)
The main theory pdf explicitly defines the training loss as KD + CE + optional regularization, and
says that this scalar loss is what produces the gradient used later by the controller.

### 5. Clean gradient exposure for later use

Even if you are not working on P/F/R logic, you should expose:
● gradients per module
● module-wise norms
● clean grouping by attention / FFN
● clean grouping by base vs LoRA if LoRA is already attached

# Best way to store the training data (A

# guideline you can follow or you can come


# up with something better :V ) -> You will

# start with this at first.

**Do not store everything in one messy text file.
Use a sample-centric schema, where each row is one training example and carries all
metadata needed for routing and distillation.**
The cleanest storage design is:

## Layer A -> Canonical sample table

This is the master dataset.
Each row should look conceptually like:
{
"sample_id": "uuid-123",
"task_type": "math_reasoning",
"domain": "math",
"subdomain": "algebra",
"difficulty": "medium",
"source": "gsm8k_like",
"input_text": "If 3x + 5 = 20, what is x?",
"target_text": "x = 5",
"has_gold_label": true,
"language": "en",
"split": "train",
"bucket_id": "math_algebra_medium"
}
This is the most important table.


Why this matters:
in theory 101 pdf, it already suggests routing can use lightweight features like **domain tag** and
retrieval cues, and repetition can be tracked by **bucket**.
So every sample should already carry:
● domain
● subdomain
● difficulty
● bucket id
● source
That makes both teacher routing and repetition tracking much easier.

## Layer B — Teacher supervision table

Do not force teacher outputs into the raw dataset itself if you want flexibility.
Store teacher outputs in a separate linked table keyed by sample_id.
Example:
{
"sample_id": "uuid-123",
"teacher_id": "math_teacher_v1",
"teacher_domain": "math",
"teacher_mode": "primary",
"teacher_output_text": "To solve 3x + 5 = 20, subtract 5 from both sides...",
"teacher_logits_path": "s3://distill/teacher_logits/uuid-123_math.npz",
"teacher_confidence": 0.93,
"teacher_entropy": 0.21,


"generated_at": "2026-03-24T10:00:00Z"
}
Why separate storage is better:
● one sample may later be run through multiple teachers
● you may want to compare math teacher vs general teacher on same sample
● logits are heavy, so better to store them in files and keep only paths in metadata
This fits your PDF directly, because each sample can have a chosen teacher or teacher mixture,
and the selected teacher distribution is part of the per-sample pipeline.

## Layer C — Training-ready view

From A + B, the training owner builds a final training row like:
{
"sample_id": "uuid-123",
"input_text": "If 3x + 5 = 20, what is x?",
"target_text": "x = 5",
"domain": "math",
"bucket_id": "math_algebra_medium",
"teacher_id": "math_teacher_v1",
"teacher_output_text": "...",
"teacher_logits_path": "...",
"route_hint": "math"
}
This is what the dataloader reads.
So:


● **raw data stays clean**
● **teacher outputs stay modular**
● **training view is assembled dynamically**
That is usually the best engineering approach.
Below I am giving all the domains we should think of now

### domain

```
● general
● math
● code
● medical
```
### task_type

Examples:
● qa
● reasoning
● solve
● codegen
● debug
● explanation


```
● summarization
● classification
```
### subdomain

Examples:
**math**
● algebra
● geometry
● calculus
**code**
● python
● algorithms
● debugging
**medical**
● symptom_reasoning
● disease_knowledge
● treatment_overview
● drug_information
● clinical_qa
● medical_explanation


**general**
● writing
● factual_qa
● instruction

### difficulty

```
● easy
● medium
● hard
```
### bucket_id

Examples:
● math_algebra_easy
● code_python_debug_medium
● medical_symptom_reasoning_hard
● medical_drug_information_medium
● general_writing_easy
What is gold label that i aforementioned?
has_gold_label means:
**does this sample have a trusted ground-truth answer written by humans or from the
dataset itself?**
So:


```
● true = there is a real target answer you trust
● false = there is no trusted target answer, so you may only have teacher-generated
supervision
```
## Example

### With gold label

#### {

"input_text": "What is 12 + 7?",
"target_text": "19",
"has_gold_label": true
}
Here the dataset already knows the correct answer is 19.

### Without gold label

#### {

"input_text": "Explain this medical paragraph in simpler words.",
"target_text": null,
"has_gold_label": false
}
Here maybe you do not have a true reference answer.
You may only use the teacher model’s output as supervision.

## Why this matters in distillation

Because your loss can be different:

### If has_gold_label = true

You can use:


```
● CE loss against the real target
● KD loss against the teacher
```
### If has_gold_label = false

Usually you only use:
● **KD loss** against the teacher
● maybe some auxiliary regularization
So this field tells the training pipeline what kind of supervision is available.


