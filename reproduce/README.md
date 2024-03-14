# Reproduce

This folder contains prompts, LLM responses and processed datasets as reported in our main experiments. It is organized as follows: 

-   `diversify-x` (Diversify X) 
    -   `gen-attr-dim` and `gen-attr-val` contain prompts and responses for attribute dimensions and attribute values generation, respectively. 
    -   `config` contains processed attribute dimensions and values. 
-   `diversify-y` (Diversify X)
    -   `gen-entity-vanilla` and `gen-entity-latent`contain prompts and responses for named entity pool generation, for the vanilla and latent variant, respectively. 
    -   `config` contains processed named entities. 
-   `sample` for NER sample generation 
    -   `gen-sample` contains prompts and responses. 
    -   `dataset` contains processed NER datasets. 
-   `correction` for LLM self-correction 
    -   `gen-correction` contains prompts and responses. 
    -   `config` contains entity-class-specific annotation instructions and demos for each dataset and each diversity approach. 
    -   `instruction&demo-pool` contains annotation instruction pool and demo pool for each entity class, shared for all diversity approaches, for illustration purposes. 
    -   `annotation-error` contains representative entity annotation errors from NER sample generation for each dataset. 
    -   `dataset` contains processed datasets with entity annotations overridden by processed corrections. 



Note 

-   LLM prompts and responses are available in 2 formats: 
    1.   A readable format, via `prompts.log` and `completion-*.txt` files, and
    2.   OpenAI API format, via `requests.jsonl` and `requests_results.jsonl` files. 
-   All folders are have date prefixes indicating date of experiments. 
-   In each processed dataset (`sample/dataset`) folder, each entity annotation triple (sentence, span, entity type) is  available in `logprobs-triple.json` files. 
-   Top-uncertain triples selected for LLM Self-Correction are available from correction generation log files (`correction/gen-correction/**/completion.log`)



