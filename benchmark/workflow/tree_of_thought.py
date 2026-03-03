from random import randint, seed

from transformers import AutoTokenizer

from minisgl.core import SamplingParams
from minisgl.workflow import WorkflowScheduler
from minisgl.frontend import PromptComponent, Node

def main():
    seed(0)

    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sampling_params = SamplingParams(temperature=1.0)
    num_branches = 2
    all_nodes = []

    # 1. plan
    question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Please generate a high-level plan for solving the following question. As the first step, just say what method and idea you will use to solve the question. You can reorganize the information in the question. Do not do the actual calculation. Keep your response concise and within 80 words. Question: {question}"}
    ]

    plan_prompt = PromptComponent(
        text=tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    )
    plan_nodes = [Node(inputs=[plan_prompt], sampling_params=sampling_params, name=f"plan-{i}") for i in range(num_branches)]
    all_nodes += plan_nodes
    
    # 2. execute
    execute_nodes = []
    for i, plan_node in enumerate(plan_nodes):
        execute_inputs = [PromptComponent(text="all", node_ref=plan_node.uid)]
        messages = [
            {"role": "user", "content": "The plan looks good! Now, use real numbers and do the calculation. Please solve the question step-by-step according to the high-level plan. Give me the final answer. Make your response short."}
        ]
        execute_inputs.append(
            PromptComponent(
                text=tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            )
        )
        execute_nodes += [Node(inputs=execute_inputs, sampling_params=sampling_params, name=f"execute-{i*num_branches+j}") for j in range(num_branches)]
    all_nodes += execute_nodes

    # 3. reflect
    reflect_nodes = []
    for i, execute_node in enumerate(execute_nodes):
        reflect_inputs = [PromptComponent(text="all", node_ref=execute_node.uid)]
        messages = [
            {"role": "user", "content": "Okay. Now, evaluate your own solution and give it a score on a scale of 1 to 5. Please do rigorous check of the correctness."}
        ]
        reflect_inputs.append(
                PromptComponent(
                text=tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            )
        )
        reflect_nodes += [Node(inputs=reflect_inputs, sampling_params=sampling_params, name=f"reflect-{i*num_branches+j}") for j in range(num_branches)]  
    all_nodes += reflect_nodes

    # 4. conclude
    conclude_nodes = []
    for i, reflect_node in enumerate(reflect_nodes):
        conclude_inputs = [PromptComponent(text="all", node_ref=reflect_node.uid)]
        messages = [
            {"role": "user", "content": "Based on your reflection, do you change your mind? Now, give me the final answer after careful consideration."}
        ]
        conclude_inputs.append(
            PromptComponent(
                text=tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            )
        )
        conclude_nodes += [Node(inputs=conclude_inputs, sampling_params=sampling_params, name=f"conclude-{i*num_branches+j}") for j in range(num_branches)]  
    all_nodes += conclude_nodes
    
    # 5. aggregate
    aggregate_inputs = [
        PromptComponent(text="generated", node_ref=conclude_node.uid) 
        for conclude_node in conclude_nodes
    ]
    aggregate_node = Node(inputs=aggregate_inputs, node_type="concatenate", name=f"aggregate")
    all_nodes.append(aggregate_node)

    # 6. Run
    workflow_scheduler = WorkflowScheduler(
        "Qwen/Qwen3-0.6B",
        schedule_policy="LPM",
        max_seq_len_override=4096,
        max_extend_tokens=16384,
        cuda_graph_max_bs=256,
        page_size=256,
    )
    results = workflow_scheduler.run_workflow(all_nodes)
    print(results[aggregate_node.uid]["text"])

if __name__ == "__main__":
    main()