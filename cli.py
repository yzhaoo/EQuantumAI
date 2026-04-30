from __future__ import annotations

import argparse
import json
import sys

from agent.schemas import AgentRuntimeConfig


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive CLI for the refactored EQuantumAI agent and simulation runner."
    )
    parser.add_argument("query", nargs="?", help="Optional first natural-language query.")
    parser.add_argument("--profile", default="dotgate_center")
    parser.add_argument("--device-shape", default="dotgate")
    parser.add_argument("--parser", default="openai", choices=["langchain", "openai", "regex"])
    parser.add_argument("--openai-model", default="gpt-4o-mini")
    parser.add_argument("--strict-openai", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--ldos-method", default="ED", choices=["TF", "ED", "kmeanssample"])
    parser.add_argument("--ncore", type=int, default=1)
    parser.add_argument("--moments", type=int, default=256)
    parser.add_argument("--n-random", type=int, default=10)
    parser.add_argument("--eta", type=float, default=0.00015)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--kernel", default="jackson")
    parser.add_argument("--energy-points", type=int, default=1024)
    parser.add_argument("--tol-poisson", type=float, default=1e-3)
    parser.add_argument("--tol-ildos", type=float, default=1e-2)
    parser.add_argument("--no-execute", action="store_true", help="Stop once the spec is ready to run.")
    parser.add_argument("--require-manual-check", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> AgentRuntimeConfig:
    return AgentRuntimeConfig(
        parser=args.parser,
        openai_model=args.openai_model,
        strict_openai=args.strict_openai,
        profile=args.profile,
        device_shape=args.device_shape,
        output_dir=args.output_dir,
        ldos_method=args.ldos_method,
        ncore=args.ncore,
        moments=args.moments,
        n_random=args.n_random,
        eta=args.eta,
        eps=args.eps,
        kernel=args.kernel,
        energy_points=args.energy_points,
        tol_poisson=args.tol_poisson,
        tol_ildos=args.tol_ildos,
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    from agent.session import AgentSession
    from simulation.runner import format_result_summary, run_agent_response

    session = AgentSession(config=config_from_args(args))
    try:
        prompt = args.query or input("Query: ").strip()
    except EOFError:
        return

    while prompt:
        response = session.turn(prompt, execute=False)
        print(json.dumps(response.to_dict(), indent=2))

        if response.status == "running":
            if args.no_execute:
                return
            completed = run_agent_response(
                response,
                config=session.config,
                status=lambda text: print(f"[status] {text}"),
                log=lambda text: print(f"[log] {text}"),
                require_manual_check=args.require_manual_check,
            )
            print(json.dumps(completed.to_dict(), indent=2))
            if completed.result:
                print(format_result_summary(completed.result))
            return

        if not sys.stdin.isatty():
            return

        try:
            prompt = input("You: ").strip()
        except EOFError:
            return


if __name__ == "__main__":
    main()
