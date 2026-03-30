import argparse
import json
import random
import time

import pandas as pd
from transformers import AutoTokenizer

from minisgl.frontend import Node, PromptComponent
from minisgl.workflow import WorkflowScheduler
from minisgl.workflow.policy import EVICT_POLICY_MAP, SCHEDULE_POLICY_MAP

from mixed_workload import (
    build_chain_workflow,
    build_deep_workflow,
    build_hybrid_workflow,
    build_wide_workflow,
    make_root_prompt,
    make_sampling_params,
    make_suffix,
    summarize_component,
)


BUILDER_MODULE = "mixed_workload_rich"


def _randint(rng, low, high):
    if low > high:
        raise ValueError(f"Invalid random range: low={low}, high={high}")
    if rng is None:
        return low
    return rng.randint(low, high)


def _sample_choices(rng, choices, weights, k):
    total = sum(weights)
    if total <= 0:
        raise ValueError("At least one weight must be positive")
    if rng is None:
        return [choices[0]] * k
    return rng.choices(choices, weights=[w / total for w in weights], k=k)


def build_bushy_workflow(wid, task, tokenizer, args, is_long, rng, spec=None):
    params = make_sampling_params(is_long, args)
    bushy_depth = spec["bushy_depth"] if spec is not None and "bushy_depth" in spec else args.bushy_depth
    bushy_branch_factor = (
        spec["bushy_branch_factor"] if spec is not None and "bushy_branch_factor" in spec else args.bushy_branch_factor
    )

    nodes = []
    root = Node(
        inputs=[make_root_prompt(tokenizer, task, "bushy_tree", is_long)],
        sampling_params=params,
        name=f"wf{wid}-bushy-root",
    )
    nodes.append(root)
    frontier = [root]

    for depth in range(1, bushy_depth + 1):
        next_frontier = []
        for parent_idx, parent in enumerate(frontier):
            for branch in range(bushy_branch_factor):
                child = Node(
                    inputs=[
                        PromptComponent(text="all", node_ref=parent.uid),
                        make_suffix(
                            f"Bushy depth {depth}, branch {branch}: expand a sub-idea or subcase from the current reasoning."
                        ),
                    ],
                    sampling_params=params,
                    name=f"wf{wid}-bushy-d{depth}-p{parent_idx}-b{branch}",
                )
                nodes.append(child)
                next_frontier.append(child)
        frontier = next_frontier

    return nodes


def build_hybrid_var_workflow(wid, task, tokenizer, args, is_long, rng, spec=None):
    params = make_sampling_params(is_long, args)
    trunk_depth = spec["hybrid_var_trunk_depth"] if spec is not None else args.hybrid_var_trunk_depth
    branches = spec["hybrid_var_branches"] if spec is not None else args.hybrid_var_branches
    branch_modes = spec["branch_modes"] if spec is not None else _sample_choices(
        rng,
        ["end", "chain", "split"],
        [args.hybrid_var_end_ratio, args.hybrid_var_chain_ratio, args.hybrid_var_split_ratio],
        branches,
    )
    branch_chain_depths = spec["branch_chain_depths"] if spec is not None else [
        _randint(rng, args.hybrid_var_chain_depth_min, args.hybrid_var_chain_depth_max) if mode == "chain" else 0
        for mode in branch_modes
    ]
    branch_split_counts = spec["branch_split_counts"] if spec is not None else [
        args.hybrid_var_rebranch_branches if mode == "split" else 0
        for mode in branch_modes
    ]

    nodes = []
    root = Node(
        inputs=[make_root_prompt(tokenizer, task, "hybrid_var_tree", is_long)],
        sampling_params=params,
        name=f"wf{wid}-hybridvar-root",
    )
    nodes.append(root)
    prev = root

    for depth in range(1, trunk_depth + 1):
        trunk = Node(
            inputs=[
                PromptComponent(text="all", node_ref=prev.uid),
                make_suffix(
                    f"Hybrid-var trunk step {depth}: build a stronger shared prefix before branches diverge."
                ),
            ],
            sampling_params=params,
            name=f"wf{wid}-hybridvar-trunk-{depth}",
        )
        nodes.append(trunk)
        prev = trunk

    for branch in range(branches):
        mode = branch_modes[branch]
        branch_node = Node(
            inputs=[
                PromptComponent(text="all", node_ref=prev.uid),
                make_suffix(
                    f"Hybrid-var branch {branch}: open a branch that may stop, continue, or split again."
                ),
            ],
            sampling_params=params,
            name=f"wf{wid}-hybridvar-branch-{branch}",
        )
        nodes.append(branch_node)

        if mode == "end":
            continue
        if mode == "chain":
            chain_prev = branch_node
            for depth in range(1, branch_chain_depths[branch] + 1):
                chain_node = Node(
                    inputs=[
                        PromptComponent(text="all", node_ref=chain_prev.uid),
                        make_suffix(
                            f"Hybrid-var branch {branch}, chain step {depth}: keep refining this branch in a linear way."
                        ),
                    ],
                    sampling_params=params,
                    name=f"wf{wid}-hybridvar-branch-{branch}-chain-{depth}",
                )
                nodes.append(chain_node)
                chain_prev = chain_node
            continue

        for split_idx in range(branch_split_counts[branch]):
            split_node = Node(
                inputs=[
                    PromptComponent(text="all", node_ref=branch_node.uid),
                    make_suffix(
                        f"Hybrid-var branch {branch}, split {split_idx}: expand this branch into a more specialized sub-branch."
                    ),
                ],
                sampling_params=params,
                name=f"wf{wid}-hybridvar-branch-{branch}-split-{split_idx}",
            )
            nodes.append(split_node)

    return nodes


def build_merge_workflow(wid, task, tokenizer, args, is_long, rng, spec=None):
    params = make_sampling_params(is_long, args)
    merge_branches = spec["merge_branches"] if spec is not None else args.merge_branches
    branch_depths = spec["merge_branch_depths"] if spec is not None else [
        _randint(rng, args.merge_branch_depth_min, args.merge_branch_depth_max)
        for _ in range(merge_branches)
    ]
    merge_refine_depth = spec["merge_refine_depth"] if spec is not None else args.merge_refine_depth

    nodes = []
    root = Node(
        inputs=[make_root_prompt(tokenizer, task, "merge_tree", is_long)],
        sampling_params=params,
        name=f"wf{wid}-merge-root",
    )
    nodes.append(root)

    merge_sources = []
    for branch in range(merge_branches):
        prev = Node(
            inputs=[
                PromptComponent(text="all", node_ref=root.uid),
                make_suffix(
                    f"Merge branch {branch}: work out one candidate solution path that may later be compared or combined."
                ),
            ],
            sampling_params=params,
            name=f"wf{wid}-merge-branch-{branch}",
        )
        nodes.append(prev)

        for depth in range(1, branch_depths[branch] + 1):
            next_node = Node(
                inputs=[
                    PromptComponent(text="all", node_ref=prev.uid),
                    make_suffix(
                        f"Merge branch {branch}, refine step {depth}: add another concrete refinement before merging."
                    ),
                ],
                sampling_params=params,
                name=f"wf{wid}-merge-branch-{branch}-refine-{depth}",
            )
            nodes.append(next_node)
            prev = next_node
        merge_sources.append(prev)

    merge_inputs = [
        PromptComponent(
            text=(
                "You are now combining several candidate branch outputs for the same task. "
                "Compare them carefully and synthesize a unified next-step answer.\n\n"
            )
        )
    ]
    for branch, source in enumerate(merge_sources):
        merge_inputs.append(PromptComponent(text=f"Candidate {branch}:\n"))
        merge_inputs.append(PromptComponent(text="generated", node_ref=source.uid))
        merge_inputs.append(PromptComponent(text="\n\n"))
    merge_inputs.append(
        PromptComponent(
            text="Synthesize the candidates into one concise merged answer that keeps the strongest ideas."
        )
    )
    merge_root = Node(
        inputs=merge_inputs,
        sampling_params=params,
        name=f"wf{wid}-merge-fanin-root",
    )
    nodes.append(merge_root)

    prev = merge_root
    for depth in range(1, merge_refine_depth + 1):
        refine = Node(
            inputs=[
                PromptComponent(text="all", node_ref=prev.uid),
                make_suffix(
                    f"Merge refine step {depth}: make the merged answer more consistent and polished."
                ),
            ],
            sampling_params=params,
            name=f"wf{wid}-merge-refine-{depth}",
        )
        nodes.append(refine)
        prev = refine

    return nodes


BUILDERS = {
    "chain": build_chain_workflow,
    "wide": build_wide_workflow,
    "deep": build_deep_workflow,
    "hybrid": build_hybrid_workflow,
    "bushy": build_bushy_workflow,
    "hybrid_var": build_hybrid_var_workflow,
    "merge": build_merge_workflow,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="model path to run")
    parser.add_argument("--data_path", type=str, default="data/gsm8k-test.parquet", help="path of the data file")
    parser.add_argument("--attention_backend", type=str, default="fi", help="attention backend")
    parser.add_argument("--num_workflows", type=int, default=12, help="number of workflow instances to generate")

    parser.add_argument("--chain_ratio", type=float, default=0.15, help="mixture ratio for chain workflows")
    parser.add_argument("--wide_ratio", type=float, default=0.15, help="mixture ratio for wide-tree workflows")
    parser.add_argument("--deep_ratio", type=float, default=0.15, help="mixture ratio for deep-tree workflows")
    parser.add_argument("--hybrid_ratio", type=float, default=0.2, help="mixture ratio for hybrid workflows")
    parser.add_argument("--bushy_ratio", type=float, default=0.15, help="mixture ratio for recursively branching workflows")
    parser.add_argument("--hybrid_var_ratio", type=float, default=0.1, help="mixture ratio for variable hybrid workflows")
    parser.add_argument("--merge_ratio", type=float, default=0.1, help="mixture ratio for branch-then-merge workflows")

    parser.add_argument("--chain_depth_min", type=int, default=2, help="minimum number of sequential child nodes in chain workflows")
    parser.add_argument("--chain_depth_max", type=int, default=6, help="maximum number of sequential child nodes in chain workflows")
    parser.add_argument("--wide_branches", type=int, default=4, help="number of first-level branches in wide-tree workflows")
    parser.add_argument("--deep_depth", type=int, default=4, help="main-chain depth in deep-tree workflows")
    parser.add_argument("--hybrid_trunk_depth", type=int, default=2, help="trunk depth before branching in hybrid workflows")
    parser.add_argument("--hybrid_branches", type=int, default=3, help="number of branches after the hybrid trunk")

    parser.add_argument("--bushy_depth", type=int, default=2, help="number of branching levels after the root in bushy workflows")
    parser.add_argument("--bushy_branch_factor", type=int, default=3, help="branching factor for bushy workflows")

    parser.add_argument("--hybrid_var_trunk_depth", type=int, default=2, help="trunk depth before variable branch behavior")
    parser.add_argument("--hybrid_var_branches", type=int, default=4, help="number of first-level branches in hybrid_var workflows")
    parser.add_argument("--hybrid_var_end_ratio", type=float, default=0.34, help="probability that a hybrid_var branch ends immediately")
    parser.add_argument("--hybrid_var_chain_ratio", type=float, default=0.33, help="probability that a hybrid_var branch continues as a chain")
    parser.add_argument("--hybrid_var_split_ratio", type=float, default=0.33, help="probability that a hybrid_var branch splits again")
    parser.add_argument("--hybrid_var_chain_depth_min", type=int, default=1, help="minimum continuation depth for chain-mode hybrid_var branches")
    parser.add_argument("--hybrid_var_chain_depth_max", type=int, default=2, help="maximum continuation depth for chain-mode hybrid_var branches")
    parser.add_argument("--hybrid_var_rebranch_branches", type=int, default=2, help="number of second-level branches for split-mode hybrid_var branches")

    parser.add_argument("--merge_branches", type=int, default=3, help="number of branches before merge in merge workflows")
    parser.add_argument("--merge_branch_depth_min", type=int, default=0, help="minimum per-branch depth before merge")
    parser.add_argument("--merge_branch_depth_max", type=int, default=1, help="maximum per-branch depth before merge")
    parser.add_argument("--merge_refine_depth", type=int, default=1, help="number of continuation nodes after the merge root")

    parser.add_argument("--long_ratio", type=float, default=0.5, help="fraction of workflows using longer prompts / outputs")
    parser.add_argument("--short_temperature", type=float, default=0.2, help="temperature for short workflows")
    parser.add_argument("--long_temperature", type=float, default=0.4, help="temperature for long workflows")
    parser.add_argument("--short_max_tokens", type=int, default=80, help="max output tokens for short workflows")
    parser.add_argument("--long_max_tokens", type=int, default=160, help="max output tokens for long workflows")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--prompt_style",
        type=str,
        default="shared_prefix",
        choices=["shared_prefix", "early_diverge"],
        help="prompt layout style: shared_prefix keeps common template first, early_diverge places task early to reduce cross-workflow prefix sharing",
    )
    parser.add_argument(
        "--random_shape_sampling",
        action="store_true",
        help="sample workflow shapes independently by ratio instead of using balanced counts",
    )
    parser.add_argument("--save_spec", type=str, default=None, help="save generated workload specification to a JSON file")
    parser.add_argument("--load_spec", type=str, default=None, help="load workload specification from a JSON file")
    parser.add_argument(
        "--schedule_policy",
        type=str,
        default=None,
        choices=sorted(SCHEDULE_POLICY_MAP.keys()),
        help="workflow schedule policy; defaults to the value currently set in prefill.py",
    )
    parser.add_argument(
        "--evict_policy",
        type=str,
        default=None,
        choices=sorted(EVICT_POLICY_MAP.keys()),
        help="workflow eviction policy; defaults to the value currently set in prefill.py",
    )
    parser.add_argument("--dry_run", action="store_true", help="only construct workflows and print shape statistics")
    parser.add_argument(
        "--verbose_dry_run",
        action="store_true",
        help="when used with --dry_run, print per-workflow and per-node prompt structure",
    )
    return parser.parse_args()


def sample_workflow_kinds(args, rng):
    weights = {
        "chain": args.chain_ratio,
        "wide": args.wide_ratio,
        "deep": args.deep_ratio,
        "hybrid": args.hybrid_ratio,
        "bushy": args.bushy_ratio,
        "hybrid_var": args.hybrid_var_ratio,
        "merge": args.merge_ratio,
    }
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("At least one workflow ratio must be positive")

    kinds = list(weights.keys())
    probs = [weights[k] / total for k in kinds]
    if args.random_shape_sampling:
        return rng.choices(kinds, weights=probs, k=args.num_workflows)

    raw_counts = [args.num_workflows * prob for prob in probs]
    counts = [int(count) for count in raw_counts]
    remainder = args.num_workflows - sum(counts)
    fractional_parts = sorted(
        [(raw_counts[i] - counts[i], i) for i in range(len(kinds))],
        key=lambda x: x[0],
        reverse=True,
    )
    for _, idx in fractional_parts[:remainder]:
        counts[idx] += 1

    workflow_kinds = []
    for kind, count in zip(kinds, counts):
        workflow_kinds.extend([kind] * count)
    rng.shuffle(workflow_kinds)
    return workflow_kinds


def generate_workflow_specs(args, tasks, rng):
    workflow_kinds = sample_workflow_kinds(args, rng)
    specs = []
    for wid, workflow_kind in enumerate(workflow_kinds):
        spec = {
            "wid": wid,
            "kind": workflow_kind,
            "task_index": wid % len(tasks),
            "is_long": rng.random() < args.long_ratio,
        }
        if workflow_kind == "chain":
            spec["chain_depth"] = _randint(rng, args.chain_depth_min, args.chain_depth_max)
        elif workflow_kind == "wide":
            spec["wide_branches"] = args.wide_branches
        elif workflow_kind == "deep":
            spec["deep_depth"] = args.deep_depth
        elif workflow_kind == "hybrid":
            spec["hybrid_trunk_depth"] = args.hybrid_trunk_depth
            spec["hybrid_branches"] = args.hybrid_branches
        elif workflow_kind == "bushy":
            spec["bushy_depth"] = args.bushy_depth
            spec["bushy_branch_factor"] = args.bushy_branch_factor
        elif workflow_kind == "hybrid_var":
            spec["hybrid_var_trunk_depth"] = args.hybrid_var_trunk_depth
            spec["hybrid_var_branches"] = args.hybrid_var_branches
            spec["branch_modes"] = _sample_choices(
                rng,
                ["end", "chain", "split"],
                [args.hybrid_var_end_ratio, args.hybrid_var_chain_ratio, args.hybrid_var_split_ratio],
                args.hybrid_var_branches,
            )
            spec["branch_chain_depths"] = [
                _randint(rng, args.hybrid_var_chain_depth_min, args.hybrid_var_chain_depth_max)
                if mode == "chain"
                else 0
                for mode in spec["branch_modes"]
            ]
            spec["branch_split_counts"] = [
                args.hybrid_var_rebranch_branches if mode == "split" else 0
                for mode in spec["branch_modes"]
            ]
        elif workflow_kind == "merge":
            spec["merge_branches"] = args.merge_branches
            spec["merge_branch_depths"] = [
                _randint(rng, args.merge_branch_depth_min, args.merge_branch_depth_max)
                for _ in range(args.merge_branches)
            ]
            spec["merge_refine_depth"] = args.merge_refine_depth
        specs.append(spec)
    return specs


def _estimate_nodes(spec):
    kind = spec["kind"]
    if kind == "chain":
        return 1 + spec["chain_depth"]
    if kind == "wide":
        return 1 + spec["wide_branches"]
    if kind == "deep":
        return 1 + 2 * spec["deep_depth"]
    if kind == "hybrid":
        return 1 + spec["hybrid_trunk_depth"] + 2 * spec["hybrid_branches"]
    if kind == "bushy":
        total = 1
        level_nodes = 1
        for _ in range(spec["bushy_depth"]):
            level_nodes *= spec["bushy_branch_factor"]
            total += level_nodes
        return total
    if kind == "hybrid_var":
        return (
            1
            + spec["hybrid_var_trunk_depth"]
            + spec["hybrid_var_branches"]
            + sum(spec["branch_chain_depths"])
            + sum(spec["branch_split_counts"])
        )
    if kind == "merge":
        return 1 + spec["merge_branches"] + sum(spec["merge_branch_depths"]) + 1 + spec["merge_refine_depth"]
    raise ValueError(f"Unknown workflow kind: {kind}")


def summarize_workflow_specs(specs):
    counts = {kind: 0 for kind in BUILDERS}
    short_count = 0
    long_count = 0
    total_nodes = 0
    for spec in specs:
        counts[spec["kind"]] += 1
        total_nodes += _estimate_nodes(spec)
        if spec["is_long"]:
            long_count += 1
        else:
            short_count += 1
    return {
        "num_workflows": len(specs),
        "estimated_num_nodes": total_nodes,
        "shape_counts": counts,
        "short_workflows": short_count,
        "long_workflows": long_count,
    }


def build_generator_config(args):
    return {
        "builder_module": BUILDER_MODULE,
        "data_path": args.data_path,
        "seed": args.seed,
        "random_shape_sampling": args.random_shape_sampling,
        "chain_ratio": args.chain_ratio,
        "wide_ratio": args.wide_ratio,
        "deep_ratio": args.deep_ratio,
        "hybrid_ratio": args.hybrid_ratio,
        "bushy_ratio": args.bushy_ratio,
        "hybrid_var_ratio": args.hybrid_var_ratio,
        "merge_ratio": args.merge_ratio,
        "chain_depth_min": args.chain_depth_min,
        "chain_depth_max": args.chain_depth_max,
        "wide_branches": args.wide_branches,
        "deep_depth": args.deep_depth,
        "hybrid_trunk_depth": args.hybrid_trunk_depth,
        "hybrid_branches": args.hybrid_branches,
        "bushy_depth": args.bushy_depth,
        "bushy_branch_factor": args.bushy_branch_factor,
        "hybrid_var_trunk_depth": args.hybrid_var_trunk_depth,
        "hybrid_var_branches": args.hybrid_var_branches,
        "hybrid_var_end_ratio": args.hybrid_var_end_ratio,
        "hybrid_var_chain_ratio": args.hybrid_var_chain_ratio,
        "hybrid_var_split_ratio": args.hybrid_var_split_ratio,
        "hybrid_var_chain_depth_min": args.hybrid_var_chain_depth_min,
        "hybrid_var_chain_depth_max": args.hybrid_var_chain_depth_max,
        "hybrid_var_rebranch_branches": args.hybrid_var_rebranch_branches,
        "merge_branches": args.merge_branches,
        "merge_branch_depth_min": args.merge_branch_depth_min,
        "merge_branch_depth_max": args.merge_branch_depth_max,
        "merge_refine_depth": args.merge_refine_depth,
        "long_ratio": args.long_ratio,
        "short_temperature": args.short_temperature,
        "long_temperature": args.long_temperature,
        "short_max_tokens": args.short_max_tokens,
        "long_max_tokens": args.long_max_tokens,
        "prompt_style": args.prompt_style,
    }


def save_workflow_specs(path, specs, args):
    payload = {
        "builder_module": BUILDER_MODULE,
        "summary": summarize_workflow_specs(specs),
        "generator_config": build_generator_config(args),
        "workflows": specs,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_workflow_specs(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    workflows = payload.get("workflows")
    if not isinstance(workflows, list):
        raise ValueError("Invalid workload spec file: missing 'workflows' list")
    return workflows


def main():
    args = parse_args()
    if args.verbose_dry_run:
        args.dry_run = True

    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer._mixed_prompt_style = args.prompt_style

    data = pd.read_parquet(args.data_path)
    tasks = data["question"].tolist()
    if len(tasks) == 0:
        raise ValueError("No tasks found in the data file")

    if args.load_spec is not None:
        workflow_specs = load_workflow_specs(args.load_spec)
    else:
        workflow_specs = generate_workflow_specs(args, tasks, rng)
        if args.save_spec is not None:
            save_workflow_specs(args.save_spec, workflow_specs, args)

    all_nodes = []
    workflow_records = []
    counts = {kind: 0 for kind in BUILDERS}
    long_count = 0
    short_count = 0

    for spec in workflow_specs:
        wid = spec["wid"]
        workflow_kind = spec["kind"]
        task_index = spec["task_index"]
        if task_index < 0 or task_index >= len(tasks):
            raise ValueError(f"Task index {task_index} out of range for data file")
        task = tasks[task_index]
        is_long = spec["is_long"]
        if is_long:
            long_count += 1
        else:
            short_count += 1

        builder = BUILDERS[workflow_kind]
        nodes = builder(wid, task, tokenizer, args, is_long, rng, spec)
        all_nodes.extend(nodes)
        workflow_records.append(
            {
                "wid": wid,
                "kind": workflow_kind,
                "is_long": is_long,
                "task": task,
                "nodes": nodes,
                "spec": spec,
            }
        )
        counts[workflow_kind] += 1

    print("Generated rich workload summary:")
    print(f"  num_workflows: {len(workflow_specs)}")
    print(f"  num_nodes: {len(all_nodes)}")
    print(f"  shape_counts: {counts}")
    print(f"  short_workflows: {short_count}")
    print(f"  long_workflows: {long_count}")
    print(f"  random_shape_sampling: {args.random_shape_sampling}")
    print(f"  prompt_style: {args.prompt_style}")
    print(f"  schedule_policy: {args.schedule_policy if args.schedule_policy is not None else 'default(from prefill.py)'}")
    print(f"  evict_policy: {args.evict_policy if args.evict_policy is not None else 'default(from prefill.py)'}")
    print(f"  load_spec: {args.load_spec if args.load_spec is not None else 'None'}")
    print(f"  save_spec: {args.save_spec if args.save_spec is not None else 'None'}")

    if args.dry_run:
        if args.verbose_dry_run:
            print("\nVerbose workflow structure:")
            for record in workflow_records:
                task_preview = record["task"].strip().replace("\n", " ")
                if len(task_preview) > 100:
                    task_preview = task_preview[:97] + "..."
                print(
                    f"\n[workflow {record['wid']}] kind={record['kind']} "
                    f"length={'long' if record['is_long'] else 'short'} "
                    f"nodes={len(record['nodes'])}"
                )
                print(f"  spec={record['spec']}")
                print(f"  task={task_preview}")
                for node in record["nodes"]:
                    component_summary = ", ".join(summarize_component(component) for component in node.inputs)
                    print(f"  - {node.name} (uid={node.uid})")
                    print(f"    inputs: [{component_summary}]")
        return

    workflow_scheduler = WorkflowScheduler(
        args.model_path,
        attention_backend=args.attention_backend,
        max_seq_len_override=8192,
        max_extend_tokens=8192,
        cuda_graph_max_bs=32,
        page_size=256,
        num_page_override=64,
        debug=True,
        schedule_policy=args.schedule_policy,
        evict_policy=args.evict_policy,
    )

    t = time.time()
    results, info = workflow_scheduler.run_workflow(all_nodes)
    t = time.time() - t
    throughput = sum(status["output_len"] for status in results.values()) / max(t, 1e-6)

    print(info)
    print(f"elapsed_time={t:.4f}s")
    print(f"throughput={throughput:.4f} tokens/s")


if __name__ == "__main__":
    main()
