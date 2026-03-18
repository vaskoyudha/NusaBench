# Supported Tasks

NusaBench includes 8 diverse tasks that cover various aspects of Indonesian Natural Language Processing. Each task is defined by a specific dataset and a set of evaluation metrics.

## Overview Table

| Task Name | Dataset | Task Type | Metrics |
|-----------|---------|-----------|---------|
| `sentiment_smsa` | `indonlp/indonlu` (smsa) | Sentiment Analysis | `accuracy`, `f1` |
| `nli_wrete` | `indonlp/indonlu` (wrete) | Natural Language Inference | `accuracy` |
| `qa_facqa` | `indonlp/indonlu` (facqa) | Question Answering | `exact_match` |
| `ner_nergrit` | `indonlp/indonlu` (nergrit) | Named Entity Recognition | `f1` |
| `summarization_indosum` | `csebuetnlp/xlsum` (indonesian) | Summarization | `rouge` |
| `mt_nusax` | `indonlp/nusa_x` (ind) | Machine Translation | `bleu`, `chrf` |
| `toxicity_id` | `indonlp/indonlu` (casa) | Toxicity Detection | `accuracy`, `f1` |
| `cultural_indommu` | `indolem/IndoMMLU` | Cultural Knowledge | `accuracy` |

---

## Task Details

### Sentiment Analysis (`sentiment_smsa`)
Evaluates the model's ability to classify the sentiment of Indonesian social media text into positive, negative, or neutral categories. Based on the SMSA dataset from IndoNLU.

### Natural Language Inference (`nli_wrete`)
Tests the model's understanding of the relationship between two sentences (premise and hypothesis), determining if the hypothesis is entailed by or contradicts the premise. Uses the WRETE dataset from IndoNLU.

### Question Answering (`qa_facqa`)
A factoid question-answering task where the model must extract the correct answer from a given context. Performance is measured using Exact Match (EM). Based on the FacQA dataset from IndoNLU.

### Named Entity Recognition (`ner_nergrit`)
Evaluates the model's ability to identify and categorize entities (Person, Organization, Location, etc.) in Indonesian text. Measured using F1-score. Uses the NERGrit dataset from IndoNLU.

### Summarization (`summarization_indosum`)
Tests the model's ability to generate concise and accurate summaries of Indonesian news articles. Evaluated using ROUGE scores. Based on the IndoSum subset of the XL-Sum dataset.

### Machine Translation (`mt_nusax`)
Evaluates translation quality from Indonesian to English (or vice versa, depending on configuration). Uses the NusaX dataset and metrics like BLEU and chrF.

### Toxicity Detection (`toxicity_id`)
A safety-focused task that identifies toxic, hateful, or abusive language in Indonesian text. Based on the CASA dataset from IndoNLU.

### Cultural Knowledge (`cultural_indommu`)
A challenging task designed to evaluate the model's understanding of Indonesian culture, history, and social norms. Based on the IndoMMLU benchmark.

---

## How Tasks are Defined

Tasks in NusaBench are defined using YAML configuration files located in `src/nusabench/tasks/configs/`. Each config specifies:
*   The HuggingFace dataset path and name.
*   The splits used for evaluation.
*   The prompt template for the LLM.
*   The metrics to be calculated.
