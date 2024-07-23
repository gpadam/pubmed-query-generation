This a release of code for our submission "Literature Search Sandbox: a Large Language Model that generates search queries for systematic reviews". 


Code:
    - [installed.txt](installed.txt): a list of installed packages; roughly equivalent to a requirements.txt; the most important parts to match are the transformers, torch, bitsandbytes, and peft releases. We expect (but have not and cannot verify) broad compatibility across similar versions of these packages, and across other packages.
    - app.py: anvil backend for running a fine-tuned model
    - [learn2query2.py](learn2query.py): fine-tuning, evaluation, decoding script
        - see the output of `--help` for all options.
        - Fine-tuning may be run as something similar to the following (on a machine with a >40GB GPU, fine-tuning was performed on either 80GB A100s or 48GB RTX8000s, and after adjusting parameters to your local settings):
            - `python learn2query.py --model_name mistralai/Mistral-7B-Instruct-v0.2 --optimizer adafactor --is_causal --do_train --resume_from_training --peft_alpha 16 --peft_r 64 --peft_dropout 0.1 --epochs 10 --lr 1e-5 --batch_size 1 --fp16 --input_field 'Review_title' --target_field 'pubmed_stripped' --data_path PROSPERO_searches_cleaned_2023.11.27..subset.csv --output_dir 'outputs/mistralai/Mistral-7B-Instruct-v0.2/10/adafactor/bs_1/lr1e-5/--peft_alpha_16_--peft_r_64_--peft_dropout_0.1/Review_title/pubmed_stripped' --generation_config generation_config.json`
        - Decoding may be run as (on a machine with a >40GB GPU, possibly less): `python learn2query.py --model_name $model --is_causal --init_lora $checkpoint_dir/ --resume_from_training  --batch_size 1 --fp16  $peft_args --input_field '$input_field'  --target_field '$target_field' --data_path ${input_file} --output_dir '${checkpoint_dir}'  --generation_config generation_config.json`
            - `model=mistralai/Mistral-7B-Instruct-v0.2`
            - `checkpoint_dir`: fine-tuning output
            - `peft_args="--peft_alpha $alpha --peft_r $_r --peft_dropout 0.1"` adjust alpha and r as necessary to match your fine-tuning configuration
            - `input_field='Review_title'` or similar field (e.g. `Review_question`)
            - `target_field='pubmed_stripped'` or generate MeSH if you desire (`pubmed_tagged`)
            - `input_file`: one of the data files below
            - `output_dir`: a directory!
    - [query_tests_function.py](query_tests_function.py): evaluation code from a jupyter notebook
    - [search_cleaning_function.R](search_cleaning_function.R): from the "search" field, isolates the search strategy, using rule-based algorithms
    - [clean_data.R](clean_data.R : apply the above to the raw PROSPERO dataset (not included here, PROPSERO should be contacted for that)
    - [generation_config.json](generation_config.json): Sample config

Data:
- [PROSPERO_searches_cleaned_2023.11.27.csv](PROSPERO_searches_cleaned_2023.11.27.csv): training/validation data used for model development
    - Fields: see below for a superset
- [PROSPERO_searches_cleaned_2023.11.27.subset.csv](PROSPERO_searches_cleaned_2023.11.27.subset.csv): data subset for model development, final training set.
    - Fields: 
        - Split: train/dev/test
        - RecordID: from PROPSERO
        - Review_title: from PROPSERO
        - Review_question: from PROSPERO
        - pubmed_tagged: query from PROSPERO, manually cleaned / correct (for syntax); complete with MeSH tags
        - pubmed_stripped: pubmed_tagged with qualifiers stripped, designed as a simplified approach
        - Review_title_question: concatenation of the fields above.
        - included_studies: known studies (for the test set, primarily)
        - simplified_title: Review_title with punctuation/spacing stripped from the start/end.
- [Wang-Adam_test_data_2023.11.27.csv](Wang-Adam_test_data_2023.11.27.csv): test set
    - Fields:
        - id: identifier
        - title: review title
        - description_questions: review questions
        - query: human pubmed query, sometimes translated.
        - included_studies: ground truth gold result, separated by commas, spaces, or both.
- [Prospero_searches_2024.01.13.csv](Prospero_searches_2024.01.13.csv): most current collected training/validation data
- [val_generated_predictions.csv](val_generated_predictions.csv): dev/val outputs from the model fine-tuned above
- [test_generated_predictions.csv](test_generated_predictions.csv): test outputs from the model fine-tuned above
