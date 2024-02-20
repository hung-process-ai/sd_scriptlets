#!/usr/bin/python3.10
### Credit to https://github.com/rockerBOO/lora-inspector/ for providing the general approach to interacting with this data.

import argparse, json, math, os, torch
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
from typing import Any, Callable, Union
from safetensors import safe_open
from torch import Tensor

def to_datetime(str: str):
    return datetime.fromtimestamp(float(str))

def parse_item(key: str, value: str) -> int | float | bool | datetime | str | None:
    if key not in schema:
        print(f"invalid key in schema {key}")
        print(value)
        return value
    if schema[key] == "int" and value == "None":
        return None
    if schema[key] == "float" and value == "None":
        return None
    if key == "ss_network_dim" and value == "Dynamic":
        return "Dynamic"
    if key == "ss_network_alpha" and value == "Dynamic":
        return "Dynamic"
    return parsers[schema[key]](value)

def search_tags(freq, token):
    """
    freq: Tag frequency
    """
    tags = []
    for k in freq.keys():
        for kitem in freq[k].keys():
            # if int(freq[k][kitem]) > 3:
            if token in kitem:
                tags.append((kitem, freq[k][kitem]))
    ordered = OrderedDict(reversed(sorted(tags, key=lambda t: t[1])))
    if len(ordered) > 0:
        return True, ordered
    else:
        return False, ordered

def check_for_tag(search_key, metadata: Union[list[dict[str, Any]], dict[str, Any]]):
    found = False
    hits = []
    if type(metadata) == list:
        for record in metadata:
            if 'ss_tag_frequency' in record.keys():
                freq = record['ss_tag_frequency']
                status, results = search_tags(json.loads(freq), search_key)
                if status:
                    found = True
                    for k, v in results.items():
                        hits.append(f"{k} ({v})")
    elif type(metadata) == dict:
        if "ss_tag_frequency" in metadata.keys():
            freq = metadata['ss_tag_frequency']
            status, results = search_tags(json.loads(freq), search_key)
            if status:
                found = True
                for k, v in results.items():
                    hits.append(f"{k} ({v})")
    return found, hits

def gather_files(rootdir):
    file_list = []
    for root, subfolder, files in os.walk(rootdir):
        for file in files:
            if 'safetensors' in file:
                file_list.append(root+'/'+file)
    return file_list

if __name__ == "__main__":
    LORA_DIR = os.path.expanduser("~/stable-diffusion/")
    parser = argparse.ArgumentParser()
    final_results = {}

    parser.add_argument(
        "tag",
        type=str,
        help="The tag to search for."
    )
    parser.add_argument(
        "-d",
        "--dir", 
        type=str, 
        help=f"Directory to search (defaults to {LORA_DIR} which is stored as LORA_DIR value in this file.)"
    )

    args = parser.parse_args()
    if args.dir:
        LORA_DIR=args.dir
    search_targets = gather_files(LORA_DIR)
    print(f"Found {len(search_targets)} files in the target directory. Beginning search for tag: {args.tag}")
    for t in search_targets:
        with safe_open(t, framework="pt", device="cpu") as f:
            result, hits = check_for_tag(args.tag, f.metadata())
        if result:
            final_results[t] = hits
    print(json.dumps(final_results, indent=2))