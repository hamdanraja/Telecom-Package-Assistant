import sys
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel

from data_loader import schema_report, inspect_report
from formatting import to_markdown
from llm import LLMUnavailable

console = Console()


def main():
    if "--schema" in sys.argv:
        print(schema_report()); return
    if "--inspect" in sys.argv:
        print(inspect_report()); return

    from orchestrator import Orchestrator
    orch = Orchestrator()
    debug = "--debug" in sys.argv
    console.print(Panel("[bold]Telecom Package Assistant[/bold] - Jazz | Zong | Ufone\n"
                        "Commands: /reset  /debug  /exit", style="cyan"))
    console.print(f"Loaded {len(orch.df)} packages: {orch.df['operator'].value_counts().to_dict()}\n")

    while True:
        try:
            q = console.input("[bold green]You:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q: continue
        if q == "/exit": break
        if q == "/reset": orch.reset(); console.print("memory cleared"); continue
        if q == "/debug": debug = not debug; console.print(f"debug={debug}"); continue

        try:
            trace = orch.run(q)
            if debug:
                console.print("PLAN: " + trace.plan.model_dump_json(exclude_defaults=True),
                              markup=False, highlight=False, soft_wrap=True, style="dim")
                if trace.comparison is not None:
                    for n in trace.comparison.notes:
                        console.print("NOTE: " + n, markup=False, highlight=False, soft_wrap=True, style="dim")
                    console.print(to_markdown(trace.comparison.table),
                                  markup=False, highlight=False, soft_wrap=True, style="dim")
            console.print("[bold blue]Assistant:[/bold blue]")
            first = True
            for tok in trace.answer_stream:
                if first and "\n|" in tok:          # the code-built tables arrive as the first chunk
                    console.print(Markdown(tok))
                else:
                    console.print(tok, end="", markup=False, highlight=False, soft_wrap=True)
                first = False
            console.print("\n")
        except LLMUnavailable as e:
            console.print(f"[red]{escape(str(e))}[/red]\n")


if __name__ == "__main__":
    main()