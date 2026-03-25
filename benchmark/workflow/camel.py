import pandas as pd
import argparse

from transformers import AutoTokenizer

from minisgl.core import SamplingParams
from minisgl.workflow import WorkflowScheduler
from minisgl.frontend import PromptComponent, Node


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="model path to run")
    parser.add_argument(
        "--attention_backend",
        type=str,
        default="fi",
        help="attention backend"
    )
    parser.add_argument("--data_path", type=str, default="data/gsm8k-test.parquet", help="path of the data file")
    parser.add_argument("--num_requests", type=int, default=5, help="number of requests")
    parser.add_argument("--num_turns", type=int, default=4, help="number of CAMEL dialogue turns")
    parser.add_argument("--task_temperature", type=float, default=0.5, help="temperature for task specifier")
    parser.add_argument("--turn_temperature", type=float, default=0.2, help="temperature for dialogue turns")
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    task_specifier_params = SamplingParams(
        temperature=args.task_temperature,
        max_tokens=80,
    )
    init_params = SamplingParams(
        temperature=args.turn_temperature,
        max_tokens=80,
    )
    turn_params = SamplingParams(
        temperature=args.turn_temperature,
        max_tokens=80,
    )

    assistant_role = "Problem Solver"
    user_role = "Problem Reviewer"

    all_nodes = []

    data = pd.read_parquet(args.data_path)
    num_requests = args.num_requests if args.num_requests is not None else len(data)

    for qid, raw_task in enumerate(data["question"].head(num_requests)):
        #print(f"Question {qid}: {raw_task}\n=============================")

        # 1. task specifier
        task_specifier_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"""You will be given a raw task.

Rewrite it into a clearer, more specific, and more actionable task for a cooperative two-agent role-playing session.

Requirements:
- Keep the original intent unchanged
- Make it concrete and executable
- Keep it concise
- Output only the rewritten task

Raw task:
{raw_task}
"""
            }
        ]

        task_specifier_prompt = PromptComponent(
            text=tokenizer.apply_chat_template(
                task_specifier_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        task_specifier_node = Node(
            inputs=[task_specifier_prompt],
            sampling_params=task_specifier_params,
            name=f"q{qid}-camel-task-specifier",
        )
        all_nodes.append(task_specifier_node)

        # 2. first CAMEL turn
        camel_turn_1_prefix = PromptComponent(
            text="""You are simulating a CAMEL-style role-playing session between two cooperative agents.

Specified task:
"""
        )

        specified_task_ref = PromptComponent(
            text="generated",
            node_ref=task_specifier_node.uid,
        )

        camel_turn_1_suffix = PromptComponent(
            text=f"""

Roles:
1. {assistant_role}
   - Responsible for producing concrete content, solutions, plans, designs, explanations, or other task-related outputs

2. {user_role}
   - Responsible for guiding the task toward successful completion by refining requirements, asking for clarification, reviewing progress, or requesting improvements

Rules:
- The two roles must cooperate to complete the specified task step by step
- Stay strictly on task
- Alternate speakers
- Each turn must add one concrete new contribution
- Do not generate more than one speaker turn
- Do not simulate future turns
- Do not generate dialogue for the other role
- Do not generate greetings, thanks, or closing remarks
- Stop immediately after completing the current speaker's single turn

The following rules apply to this turn and all later turns:
- The sentence must start with the speaker name followed by a colon
- The sentence must make one concrete contribution relevant to the specified task
- Do not generate dialogue for the other role
- Do not simulate future turns
- Do not generate greetings, thanks, or closing remarks
- Stop immediately after that one sentence

Start the role-playing session now.

Next speaker: {user_role}
The sentence must start with "{user_role}:"
"""
        )

        turn_1_node = Node(
            inputs=[
                camel_turn_1_prefix,
                specified_task_ref,
                camel_turn_1_suffix,
            ],
            sampling_params=init_params,
            name=f"q{qid}-camel-turn-1",
        )
        all_nodes.append(turn_1_node)

        prev_node = turn_1_node

        # 3. later turns
        for turn_idx in range(2, args.num_turns + 1):
            speaker = assistant_role if turn_idx % 2 == 0 else user_role

            next_turn_instruction = PromptComponent(
                text=(
                    f'\n\nNext speaker: {speaker}\n'
                    f'The sentence must start with "{speaker}:"\n'
                )
            )

            turn_node = Node(
                inputs=[
                    PromptComponent(text="all", node_ref=prev_node.uid),
                    next_turn_instruction,
                ],
                sampling_params=turn_params,
                name=f"q{qid}-camel-turn-{turn_idx}",
            )
            all_nodes.append(turn_node)
            prev_node = turn_node

        ## 4. inspect final chain content for this question
        #inspect_node = Node(
        #    inputs=[
        #        PromptComponent(text="all", node_ref=prev_node.uid),
        #    ],
        #    node_type="concatenate",
        #    name=f"q{qid}-camel-inspect-final",
        #)
        #all_nodes.append(inspect_node)

    # Run
    workflow_scheduler = WorkflowScheduler(
        args.model_path,
        attention_backend=args.attention_backend,
        max_seq_len_override=8192,
        max_extend_tokens=16384,
        cuda_graph_max_bs=32,
        page_size=256,
        debug=True,
    )

    results, info = workflow_scheduler.run_workflow(all_nodes)
    print(info)
    #print(results[turn_node.uid]["text"])


if __name__ == "__main__":
    main()