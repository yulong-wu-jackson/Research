take a look of the research idea in    

Dive deep into details and understand what it want to do, and learn about the current feed back and potential concerns

Then you should conduct a deep research your self regarding this topic and conduct a well defined review on this idea inorder to make it into a real doable research that have a ability to be published in top venue in 2026

You should verify all the infomation and ensure you conduct a well defined research

you need to output a research proposal/idea doc as result, this doc should be clear and concise and well designed, and make sure infomrmation inside is accurate, is doc should generated at the research-proposal folder

dive deep and try to understand how it is doing and then do deep research on it to see if this research idea is really valid and have the potential to publish on 2026 top academic conference

You should AskUserQuestion if you need to clarify anything during the deep research, this will be a long run and complex task, you should make lots of effort on it.

I don't need a long report, but I want you to dive deep and get and analyze important information with maximum effort and give out a clear and concise and well design output (DON'T MAKE the result doc LONG!)


# review prompt
Here is a research idea that I drafted, do deep dive research on this topic, look into details and understand what it want to do, and learn about the current feed back and potential concerns, to see if this research idea is really valid and have the potential to publish on 2026 top academic conference
You should verify all the infomation and ensure you conduct a well defined research
your goal is: 
1. Do deep dive research on this topic, look into details and understand what it want to do
2. Learn about the current feed back and potential concerns, to see if this research idea is really valid and analyze the potential to publish on 2026 top academic conference
3. Give actionable feedback on various aspect: from research goal/direction to implentation
This will be a long run and complex task, you should make lots of effort on it, should continue with researching with no stop until you verified all info and have well understanding on the topic and have a in depth review on it

# code review prompt

Deep review of the UniMoE "Kill Gate" experiment codebase. This is a research project that fine-tunes Qwen3-0.6B with LoRA for joint embedding + reranking, measures task interference (TIR), and decides whether to proceed with MoE-LoRA. Wrong results here mean wrong paper conclusions. Review the FULL codebase: src/unimoe/**, configs/**, scripts/**, modal_app.py, tests/**.

Investigate the following aspects, prioritized by impact on paper validity:

1. **Scientific Correctness** — Are loss functions (InfoNCE, RerankingSFT), evaluation metrics (TIR, nDCG@10, MRR@10), significance tests (bootstrap), and kill gate verdict logic mathematically correct? Any data leakage between train and eval?

2. **Training Correctness** — Does joint alternating-batch training work correctly with gradient accumulation? Does gradient conflict measurement contaminate training state? Are LR schedule, seed setting, and loss weight balancing implemented properly?

3. **Data Pipeline** — Is hard negative sampling avoiding false negatives? Are tokenization, padding, and collation correct for a decoder-only model? Could cached datasets cause cross-seed contamination?

4. **Model Architecture** — Is last-token pooling correct for left-padded causal models? Are base params frozen and only LoRA trainable? Are yes/no token IDs resolved correctly?

5. **Memory & Performance** — Are large data structures released after use? Is GPU memory cleaned between training and evaluation? Could batch sizes cause OOM on A100-40GB?

6. **Configuration & Reproducibility** — Are the 6 YAML configs internally consistent? Could parallel Modal runs overwrite each other? Is volume commit/reload pattern correct for data consistency?

7. **Test Coverage** — Are there critical code paths not tested? Do tests verify numerical correctness or just "doesn't crash"?

Report by severity: CRITICAL (wrong results) > HIGH (training fails) > MEDIUM (suboptimal) > LOW (style). Include file:line, what's wrong, and why it matters for the paper.

