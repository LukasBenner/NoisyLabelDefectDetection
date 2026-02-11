#!/usr/bin/env python3
"""
ComfyUI batch generator using:
  - a full prompt template file (prompt_base.txt)
  - a defect block file (defects/*.txt)
and injecting the defect via {{DEFECT_BLOCK}}.

Nodes assumed (default IDs match your example):
  16 = DPRandomGenerator (inputs.text, inputs.seed)
  17 = Image Saver Simple (inputs.path, inputs.filename)
  10 = RandomNoise (inputs.noise_seed)

Install:
  pip install requests

Usage:
  python comfy_defect_batch2.py \
    --template ./workflow_flux2.json \
    --prompt_template_file ./prompt_base.txt \
    --defect_name heat_tint \
    --defect_block_file ./defects/heat_tint.txt \
    --out_base /home/lukasb/Pictures/Comfy \
    --count 200
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


# -------------------------
# ComfyUI API client
# -------------------------

@dataclass
class ComfyClient:
    server: str = "http://127.0.0.1:8188"
    timeout_s: int = 60

    def _url(self, path: str) -> str:
        return self.server.rstrip("/") + path

    def healthcheck(self) -> None:
        try:
            r = requests.get(self._url("/system_stats"), timeout=10)
            if r.ok:
                return
        except requests.RequestException:
            pass
        r = requests.get(self._url("/"), timeout=10)
        r.raise_for_status()

    def queue_prompt(self, workflow: Dict[str, Any], client_id: str = "defect_batch") -> str:
        payload = {"prompt": workflow, "client_id": client_id}
        r = requests.post(self._url("/prompt"), json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        js = r.json()
        if "prompt_id" not in js:
            raise RuntimeError(f"Unexpected /prompt response: {js}")
        return js["prompt_id"]

    def get_history(self, prompt_id: str) -> Dict[str, Any]:
        r = requests.get(self._url(f"/history/{prompt_id}"), timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def wait_done(self, prompt_id: str, poll_s: float = 0.5, max_wait_s: float = 600.0) -> Dict[str, Any]:
        t0 = time.time()
        while True:
            hist = self.get_history(prompt_id)
            item = hist.get(prompt_id)
            if isinstance(item, dict) and item.get("outputs") is not None:
                return item
            if time.time() - t0 > max_wait_s:
                raise TimeoutError(f"Timed out waiting for prompt {prompt_id} after {max_wait_s}s")
            time.sleep(poll_s)


# -------------------------
# Helpers
# -------------------------

def slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "defect"


def build_prompt_from_files(prompt_template: str, defect_block: str) -> str:
    if "{{DEFECT_BLOCK}}" not in prompt_template:
        raise ValueError("Prompt template is missing the {{DEFECT_BLOCK}} placeholder.")
    # Keep formatting clean: ensure defect block ends with a blank line
    defect = defect_block.rstrip() + "\n"
    return prompt_template.replace("{{DEFECT_BLOCK}}", defect)


def patch_workflow(
    wf: Dict[str, Any],
    *,
    full_prompt: str,
    defect_name: str,
    out_base: Path,
    prompt_node_id: str = "16",
    saver_node_id: str = "17",
    noise_node_id: str = "10",
    prompt_seed_node_id: str = "16",
    set_noise_seed: Optional[int] = None,
    set_prompt_seed: Optional[int] = None,
) -> Dict[str, Any]:
    wf2 = json.loads(json.dumps(wf))
    dslug = slugify(defect_name)

    # Patch prompt
    try:
        wf2[prompt_node_id]["inputs"]["text"] = full_prompt
    except KeyError as e:
        raise KeyError(f"Missing node {prompt_node_id} or its inputs.text. Error: {e}") from e

    # Patch output path + filename
    out_dir = out_base / dslug
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        wf2[saver_node_id]["inputs"]["path"] = str(out_dir)
        wf2[saver_node_id]["inputs"]["filename"] = f"{dslug}_syn_%time"
    except KeyError as e:
        raise KeyError(f"Missing saver node {saver_node_id} inputs.path/filename. Error: {e}") from e

    # Seeds (optional)
    if set_noise_seed is not None:
        try:
            wf2[noise_node_id]["inputs"]["noise_seed"] = int(set_noise_seed)
        except KeyError as e:
            raise KeyError(f"Missing noise node {noise_node_id} inputs.noise_seed. Error: {e}") from e

    if set_prompt_seed is not None:
        try:
            wf2[prompt_seed_node_id]["inputs"]["seed"] = int(set_prompt_seed)
        except KeyError as e:
            raise KeyError(f"Missing prompt seed node {prompt_seed_node_id} inputs.seed. Error: {e}") from e

    return wf2


def load_defects(defects_json: str, defect_name: str, defect_block_file: str) -> List[Dict[str, str]]:
    if defects_json:
        data = json.loads(Path(defects_json).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("--defects_json must be a JSON array of {name, block_file} or {name, block}")
        out: List[Dict[str, str]] = []
        for item in data:
            if not isinstance(item, dict) or "name" not in item:
                raise ValueError("Each defects_json item must have at least 'name'")
            name = str(item["name"])
            if "block" in item:
                block = str(item["block"])
            elif "block_file" in item:
                block = Path(item["block_file"]).read_text(encoding="utf-8")
            else:
                raise ValueError("Each defects_json item must include 'block' or 'block_file'")
            out.append({"name": name, "block": block})
        return out

    if not defect_name or not defect_block_file:
        raise ValueError("For single defect: provide --defect_name and --defect_block_file (or use --defects_json).")
    block = Path(defect_block_file).read_text(encoding="utf-8")
    return [{"name": defect_name, "block": block}]


@dataclass
class ProgressBar:
    total: int
    width: int = 30
    stream: Any = sys.stdout
    _done: int = 0
    _last_len: int = 0
    _start_time: float = time.time()

    def update(self, message: str = "") -> None:
        self._done += 1
        total = max(self.total, 1)
        filled = int(self.width * self._done / total)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = max(time.time() - self._start_time, 1e-6)
        avg = elapsed / self._done
        remaining = max(self.total - self._done, 0)
        eta = remaining * avg
        line = f"[{bar}] {self._done}/{self.total}"
        line += f" | elapsed {elapsed:6.1f}s avg {avg:5.2f}s ETA {eta:6.1f}s"
        if message:
            line += f" {message}"
        padded = line + " " * max(0, self._last_len - len(line))
        self.stream.write("\r" + padded)
        self.stream.flush()
        self._last_len = len(line)

    def finish(self) -> None:
        if self._last_len:
            self.stream.write("\n")
            self.stream.flush()


# -------------------------
# Main
# -------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch-generate synthetic defects via ComfyUI template workflow patching")
    p.add_argument("--server", default="http://127.0.0.1:8188")
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--max_wait", type=float, default=600.0)

    p.add_argument("--template", required=True, help="Workflow JSON template (your example saved to file)")
    p.add_argument("--prompt_template_file", required=True, help="Text file containing the full prompt template")
    p.add_argument("--out_base", required=True, help="Base output directory; subfolder per defect will be created")
    p.add_argument("--count", type=int, default=10, help="Images per defect")

    # Defect sources
    p.add_argument("--defects_json", default="", help="JSON list of defects (optional)")
    p.add_argument("--defect_name", default="", help="Single defect name (if not using defects_json)")
    p.add_argument("--defect_block_file", default="", help="Single defect block file (if not using defects_json)")

    # Node IDs (defaults match your workflow)
    p.add_argument("--prompt_node_id", default="16")
    p.add_argument("--saver_node_id", default="17")
    p.add_argument("--noise_node_id", default="10")
    p.add_argument("--prompt_seed_node_id", default="16")

    # Seeding
    p.add_argument("--seed_base", type=int, default=None, help="If set: uses seed_base+i for each generated image")
    p.add_argument("--rng_seed", type=int, default=1337, help="RNG seed if seed_base not set")

    return p


def main() -> None:
    args = build_argparser().parse_args()

    wf_template = json.loads(Path(args.template).read_text(encoding="utf-8"))
    prompt_template = Path(args.prompt_template_file).read_text(encoding="utf-8")

    out_base = Path(args.out_base).expanduser().resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    defects = load_defects(args.defects_json, args.defect_name, args.defect_block_file)

    rng = random.Random(args.rng_seed)

    client = ComfyClient(server=args.server, timeout_s=args.timeout)
    client.healthcheck()

    total = len(defects) * args.count
    progress = ProgressBar(total=total)

    for d in defects:
        for i in range(args.count):
            seed = (args.seed_base + i) if args.seed_base is not None else rng.randrange(0, 2**31 - 1)
            full_prompt = build_prompt_from_files(prompt_template, d["block"])

            wf_run = patch_workflow(
                wf_template,
                full_prompt=full_prompt,
                defect_name=d["name"],
                out_base=out_base,
                prompt_node_id=args.prompt_node_id,
                saver_node_id=args.saver_node_id,
                noise_node_id=args.noise_node_id,
                prompt_seed_node_id=args.prompt_seed_node_id,
                set_noise_seed=seed,
                set_prompt_seed=seed,
            )

            prompt_id = client.queue_prompt(wf_run, client_id="defect_batch")
            client.wait_done(prompt_id, max_wait_s=args.max_wait)

            progress.update(message=f"defect={d['name']} seed={seed} prompt_id={prompt_id}")

    progress.finish()
    print("All done.")


if __name__ == "__main__":
    main()
