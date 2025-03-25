from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

# Load model and tokenizer
model_path = "RUC-AIBOX/STILL-2"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# PROMPT
PROMPT = 'Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process.\n\nPlease structure your response into two main sections: Thought and Solution.\n\nIn the Thought section, detail your reasoning process using the specified format:\n\n```\n<|begin_of_thought|>\n{thought with steps seperated with "\n\n"}\n<|end_of_thought|>\n```\n\nEach step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. Try to use casual, genuine phrases like: "Hmm...", "This is interesting because...", "Wait, let me think about...", "Actually...", "Now that I look at it...", "This reminds me of...", "I wonder if...", "But then again...", "Let\'s see if...", "Alternatively...", "Let\'s summaize existing information...", "This might mean that...", "why/how/when/where...", etc, to make your thought process be coherent, clear, and logically sound, effectively simulating human cognitive processes.\n\nIn the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows:\n\n```\n<|begin_of_solution|>\n{final formatted, precise, and clear solution}\n<|end_of_solution|>\n```\n\nNow, try to solve the following question through the above guidlines:\n'

# Input text
question = "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$"

input_prompts = tokenizer.apply_chat_template(
    [{"role": "user", "content": PROMPT + question}],
    tokenize=False,
    add_generation_prompt=True,
)

# Params
stop_words = ["<|im_end|>", "<|endoftext|>"]

llm = LLM(
    model=model_path,
    tensor_parallel_size=8,
    max_model_len=int(1.5 * 20000),
    gpu_memory_utilization=0.95,
    dtype="bfloat16",
)

sampling_params = SamplingParams(
    temperature=0,
    top_p=1.0,
    max_tokens=20000,
    stop=stop_words,
    seed=42,
    skip_special_tokens=False,
)

# Completion
responses = llm.generate(input_prompts, sampling_params)
print(responses[0].outputs[0].text)