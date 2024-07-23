import anvil.server
anvil.server.connect("server_CMVXBN5XD4GAVVPZAVFB5SJL-2H46ZMUQS5LNG5IU")

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
import time
import torch

loading_start = time.time()
print('loading tokenizer')
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')

dtype=torch.float32
print('loading base model')
#base_model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2', torch_dtype=dtype)
print('loading title only model')
#title_only_model = AutoPeftModelForCausalLM.from_pretrained(base_model, './Review_title', is_trainable=False, torch_dtype=dtype).merge_and_unload()
title_only_model = AutoPeftModelForCausalLM.from_pretrained('Review_title', is_trainable=False, torch_dtype=dtype).merge_and_unload()
title_only_model.eval()
print('loading question only model')
#questions_only_model = AutoPeftModelForCausalLM.from_pretrained('Review_question', is_trainable=False, torch_dtype=dtype).merge_and_unload()
#questions_only_model.eval()
print('loading title questions model')
#title_questions_model = AutoPeftModelForCausalLM.from_pretrained(base_model, './Review_title_question', is_trainable=False, torch_dtype=dtype).merge_and_unload()
title_questions_model = AutoPeftModelForCausalLM.from_pretrained('Review_title_question', is_trainable=False, torch_dtype=dtype).merge_and_unload()
title_questions_model.eval()
# TODO consider loading/unloading adapters for speed/gpu
loading_end = time.time()
print('loading models took', loading_end - loading_start)

@anvil.server.callable
def query_generator(title, KQ):
    print('title', title)
    print('KQ', KQ)
    start1 = time.time()
    title_inputs = tokenizer.apply_chat_template([{'role': 'user', 'content': f'Translate the following into a boolean search query to find relevant studies in PubMed. Do not add any explanation. Do not repeat terms. Prefer shorter queries.\n{title}'}], return_tensors='pt')
    titlequery = title_only_model.generate(input_ids=title_inputs, min_length=5, no_repeat_ngram_size=5, pad_token_id=tokenizer.eos_token_id, max_new_tokens=2048)
    titlequery = tokenizer.decode(titlequery.squeeze())
    titlequery = titlequery.split('[/INST]')[-1]
    titlequery = titlequery.replace('</s>', '')
    start2 = time.time()
    print('title query', titlequery)
    print('title only duration', start2 - start1)

    kqquery = "Not trained yet"
    #question_inputs = tokenizer.apply_chat_template([{'role': 'user', 'content': f'Translate the following into a boolean search query to find relevant studies in PubMed. Do not add any explanation. Do not repeat terms. Prefer shorter queries.\n {KQ}'}], return_tensors='pt')
    #kqquery = questions_only_model.generate(input_ids=question_inputs, min_length=5, no_repeat_ngram_size=5, pad_token_id=tokenizer.eos_token_id, max_new_tokens=2048)
    #kqquery = tokenizer.decode(kqquery.squeeze())
    #kqquery = kqquery.split('[/INST]')[-1]
    #kqquery = kqquery.replace('</s>', '')
    start3 = time.time()
    print('questions query', kqquery)
    print('questions duration', start3 - start2)

    title_question_inputs = tokenizer.apply_chat_template([{'role': 'user', 'content': f'Translate the following into a boolean search query to find relevant studies in PubMed. Do not add any explanation. Do not repeat terms. Prefer shorter queries.\nReview title: \"{title}\"; Review question: {KQ}'}], return_tensors='pt')
    TIKQ = title_questions_model.generate(input_ids=title_question_inputs, min_length=5, no_repeat_ngram_size=5, pad_token_id=tokenizer.eos_token_id, max_new_tokens=2048)
    TIKQ = tokenizer.decode(TIKQ.squeeze())
    TIKQ = TIKQ.split('[/INST]')[-1]
    TIKQ = TIKQ.replace('</s>', '')
    end = time.time()
    print('title question query', TIKQ)
    print('title question duration', end - start3)

    #return (titlequery, TIKQ)
    return (titlequery, kqquery, TIKQ)

sample_title = 'Opioid Treatments for Chronic Pain'
sample_question = """
Key Question 1. Effectiveness and Comparative Effectiveness a. In patients with chronic pain, what is the effectiveness of opioids versus placebo or no opioid for outcomes related to pain, function, and quality of\nlife after short-term followup (1 to <6 months), intermediate-term followup (6 to <12 months), and long-term followup (≥12 months)? 3 b. How does effectiveness vary depending on: (1) the specific type or\ncause of pain (e.g., neuropathic, musculoskeletal [including low back pain], visceral pain, fibromyalgia, sickle cell disease, inflammatory pain, headache disorders, and degree of nociplasticity); (2) patient demographics (e.g., age, race, ethnicity, gender, socioeconomic status); (3) patient comorbidities (including past or current alcohol or substance use disorders, mental health disorders, medical comorbidities, and high risk for opioid use disorder); (4) the mechanism of action of opioids used (e.g., pure opioid agonists, partial opioid agonists such as buprenorphine, or drugs with mixed opioid and nonopioid mechanisms of action such as tramadol or tapentadol)? c. In patients with chronic pain, what is the comparative effectiveness of opioids versus nonopioid therapies (pharmacologic or nonpharmacologic, including cannabis) on outcomes related to pain, function, and quality of life after short-term followup (1 to <6 months), intermediate-term followup (6 to <12 months), and long-term followup (≥12 months)?\nd. In patients with chronic pain, what is the comparative effectiveness of opioids plus nonopioid interventions (pharmacologic or nonpharmacologic, including cannabis) versus opioids or nonopioid interventions alone on outcomes related to pain, function, quality of life, and doses of opioids used after short-term followup (1 to <6 months), intermediate-term followup (6 to <12 months), and long-term followup (≥12 months)? Key Question 2. Harms and Adverse Events a. In patients with chronic pain, what are the risks of opioids versus placebo or no opioid on: (1) opioid use disorder, abuse, or misuse; (2) overdose (intentional and unintentional); and (3) other harms, including gastrointestinal-related harms, falls, fractures, motor vehicle accidents, endocrinological harms, infections, cardiovascular events, cognitive harms,\nand psychological harms (e.g., depression)?  4 b. How do harms vary depending on: (1) the specific type or cause of pain (e.g., neuropathic, musculoskeletal [including low back pain], visceral pain, fibromyalgia, sickle cell disease, inflammatory pain, headache disorders, and degree of nociplasticity); (2) patient demographics; (3) patient comorbidities (including past or current opioid use disorder or at high risk for opioid use disorder); (4) the dose of opioids used and duration of therapy; (5) the mechanism of action of opioids used (e.g., pure opioid agonists, partial opioid agonists such as buprenorphine, or drugs with opioid and nonopioid mechanisms of action such as tramadol and tapentadol); (6) use of sedative hypnotics; (7) use of gabapentinoids; (8) use of cannabis? c. In patients with chronic pain, what are the comparative risks of opioids versus nonopioid therapies on: (1) opioid use disorder, abuse, or misuse; (2) overdose (intentional and unintentional); and (3) other harms, including gastrointestinal-related harms, falls, fractures, motor vehicle accidents, endocrinological harms, infections, cardiovascular events, cognitive harms, and mental health harms (e.g., depression)? d. In patients with chronic pain, what are the comparative risks of opioids plus nonopioid interventions (pharmacologic or nonpharmacologic, including cannabis) versus opioids or nonopioid interventions alone on: (1) opioid use disorder, abuse, or misuse; (2) overdose (intentional and unintentional); and (3) other harms, including gastrointestinal-related harms, falls, fractures, motor vehicle accidents, endocrinological harms, infections, cardiovascular events, cognitive harms, and mental health harms (e.g., depression)? Key Question 3. Dosing Strategies a. In patients with chronic pain, what is the comparative effectiveness of different methods for initiating and titrating opioids for outcomes related to pain, function, and quality of life; risk of opioid use disorder, abuse, or misuse; overdose; and doses of opioids used? b. In patients with chronic pain, what is the comparative effectiveness of short-acting versus long-acting opioids on outcomes related to pain, function, and quality of life; risk of opioid use disorder, abuse, or misuse; overdose; and doses of opioids used?  5 c. In patients with chronic pain, what is the comparative effectiveness of different long-acting opioids on outcomes related to pain, function, and quality of life; risk of opioid use disorder, abuse, or misuse; and overdose? d. In patients with chronic pain, what is the comparative effectiveness of short- plus long-acting opioids versus long-acting opioids alone on outcomes related to pain, function, and quality of life; risk of opioid use disorder, abuse, or misuse; overdose; and doses of opioids used? e. In patients with chronic pain, what is the comparative effectiveness ofscheduled, continuous versus as-needed dosing of opioids on outcomes related to pain, function, and quality of life; risk of opioid use disorder, abuse, or misuse; overdose; and doses of opioids used? f. In patients with chronic pain, what is the comparative effectiveness of opioid dose escalation versus dose maintenance or use of dose thresholds on outcomes related to pain, function, and quality of life? g. In patients with chronic pain, what is the comparative effectiveness of opioid rotation versus maintenance of current opioid therapy on outcomes related to pain, function, and quality of life, and doses of opioids used? h. In patients with chronic pain, what is the comparative effectiveness of different strategies for treating acute exacerbations of chronic pain on outcomes related to pain, function, and quality of life? i. In patients with chronic pain, what are the effects of decreasing opioid doses or of tapering off opioids versus continuation of opioids on outcomes related to pain, function, quality of life, and opiate withdrawal symptoms?. In patients with chronic pain, what is the comparative effectiveness of different tapering protocols and strategies on measures related to pain function, quality of life, opiate withdrawal symptoms, and likelihood of opioid cessation? k. In patients with chronic pain, what is the comparative effectiveness of different opioid dosages and durations of therapy for outcomes related to pain, function, and quality of life? Key Question 4. Risk Assessment and Risk Mitigation Strategies a. In patients with chronic pain being considered for opioid therapy, what is the accuracy of instruments and tests (including metabolic and/or genetic testing) for predicting risk of opioid use disorder, abuse, or misuse, and overdose? b. In patients with chronic pain, what is the effectiveness of use of risk prediction instruments and tests (including metabolic and/or genetic testing) on outcomes related to opioid use disorder, abuse, or misuse, and overdose? c. In patients with chronic pain who are prescribed opioid therapy, what is the effectiveness of risk mitigation strategies, including (1) opioid management plans, (2) patient education, (3) urine drug screening, (4) use of prescription drug monitoring program data, (5) use of monitoring instruments, (6) more frequent monitoring intervals, (7) pill counts, (8) use of abuse-deterrent formulations, (9) consultation with mental health providers when mental health conditions are present, (10) avoidance of coprescribing of sedative hypnotics, and (11) co-prescribing of naloxone on outcomes related to opioid use disorder, abuse, or misuse, and overdose? d. In patients with chronic pain, what is the comparative effectiveness of treatment strategies for managing patients with opioid use disorder related to prescription opioids on outcomes related to pain, function, quality of life, opioid use disorder, abuse, misuse, and overdose?
"""
query_generator(sample_title, sample_question)
anvil.server.wait_forever()

