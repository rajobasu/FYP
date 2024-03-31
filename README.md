ist of research papers used.

[1] https://arxiv.org/pdf/2402.05467.pdf   
[2] https://arxiv.org/pdf/2310.08419.pdf  
[3] https://arxiv.org/pdf/2305.03495.pdf   
[4] https://arxiv.org/pdf/2402.15727.pdf SelfDefend Paper  
[5] https://arxiv.org/pdf/2402.13457.pdf   Comprehensive Study   
[6] https://arxiv.org/pdf/2402.05668.pdf Survey study    
[7] https://proceedings.neurips.cc/paper_files/paper/2023/file/fd6613131889a4b656206c50a8bd7790-Paper-Conference.pdf
Important paper, discusses why attacks succeed   
[8] https://arxiv.org/pdf/2402.08679.pdf Uses Controllable text generation. best for morpher   
[9] https://arxiv.org/pdf/2402.09091.pdf Weird paper, breaks down prompt into secondary prompts which are indirect clues
to actual intent.    
[10] https://arxiv.org/pdf/2402.02309.pdf against MLLM    
[11] https://arxiv.org/pdf/2401.06824.pdf white box attacks using representation spaces.    
[12] https://openreview.net/pdf?id=1zt8GWZ9sc Using a lot of LLMs to do role play   
[13] https://arxiv.org/pdf/2402.09177.pdf Instead of directly asking prompt, make some interactive questions which leads
up to it.    
[14] https://arxiv.org/pdf/2309.10253.pdf    
[15] https://hal.science/hal-04328468v1/document    
[16] https://arxiv.org/pdf/2311.07689.pdf MART finetuned model  
[17] https://arxiv.org/pdf/2310.08419.pdf   
[18] https://arxiv.org/pdf/2307.08715.pdf   
[19] https://arxiv.org/pdf/2309.01446.pdf   
[20] https://arxiv.org/pdf/2311.03191.pdf    
[21] https://github.com/tatsu-lab/alpaca_eval Evaluation software   
[22] https://arxiv.org/pdf/2310.04451.pdf   
[23] https://arxiv.org/html/2312.02119v2    
[24] https://proceedings.neurips.cc/paper_files/paper/2022/file/3e25d1aff47964c8409fd5c8dc0438d7-Paper-Conference.pdf
COLD Model    
[25] Safety classifier

Ideas:

- binary search on the cost function.

----------


synonym sub

- https://github.com/Samsung/LexSubGen
- https://github.com/26hzhang/ConceptualPrimitives/blob/master/main.py
- https://github.com/alicediakova/Lexical-Substitution/blob/main/nlp-hw4/lexsub_main.py#L147

toxicity alternative models:

- https://huggingface.co/s-nlp/roberta_toxicity_classifier?text=This+is+such+a+bad+thing+to+do

Conclusions:

- the number of iterations done to see how much time it takes for random morphing to converge also depends on the
  scoring function
- We get more information from the distance function for harder models [61, 62, 63]