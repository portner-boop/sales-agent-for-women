import argparse
import json

from app.modules.agent.policy import POLICY_DIRECTORY, PolicyCatalog


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = POLICY_DIRECTORY / "manifest.json"
    rendered = (
        json.dumps(PolicyCatalog.load().manifest(), ensure_ascii=False, indent=2) + "\n"
    )
    if args.check:
        if path.read_text() != rendered:
            raise SystemExit("Policy manifest is stale")
    else:
        path.write_text(rendered)


if __name__ == "__main__":
    main()
