#!/usr/bin/env python3

# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import ast
import os
import re
import argparse
import logging
from typing import Dict, List, Tuple, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Compare equations defined in this repo
LAYERS_FILE_DEFAULT = os.path.join(ROOT, 'src', 'nerva_jax', 'layers.py')
# SymPy tests live in the nerva-sympy repository; default relative location
SYMPY_REPO_DEFAULT = os.path.join(ROOT, '..', 'nerva-sympy')

# Variable names we consider as part of equations to compare
EQ_VARS = {
    'feedforward': {'R', 'Sigma', 'inv_sqrt_Sigma', 'Z', 'Y'},
    'backpropagation': {'DZ', 'DW', 'Db', 'DX', 'Dbeta', 'Dgamma', 'Dal', 'Dar', 'Dtl', 'Dtr'},
}
# Map attribute names on self to variable names for comparison
ATTR_TO_VAR = {name: name for name in (EQ_VARS['feedforward'] | EQ_VARS['backpropagation'])}

# Map layer class names to a simple key used during comparison
LAYER_KEYS = {
    'LinearLayer': 'LinearLayer',
    'ActivationLayer': 'ActivationLayer',
    'SReLULayer': 'SReLULayer',
    'SoftmaxLayer': 'SoftmaxLayer',
    'LogSoftmaxLayer': 'LogSoftmaxLayer',
    'BatchNormalizationLayer': 'BatchNormalizationLayer',
}

# A normalization function for expressions to compare textual equality robustly
_ws_re = re.compile(r"\s+")

def norm_expr(s: str) -> str:
    # strip comments and spaces
    s = s.split('#', 1)[0]
    s = s.strip()
    s = _ws_re.sub(' ', s)
    # normalize some spacing artifacts
    s = s.replace(' .T', '.T')
    # normalize matmul '@' to '*' used in sympy tests
    s = s.replace('@', '*')
    # normalize parentheses around RHS to avoid trivial mismatches
    s = re.sub(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\((.*)\)$", r"\1 = \2", s)
    return s


def extract_method_equations_from_def(src: str, func_def: Optional[ast.FunctionDef]) -> List[Tuple[str, str]]:
    ordered: List[Tuple[str, str]] = []
    if func_def is None:
        return ordered
    for stmt in func_def.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            if isinstance(target, ast.Name):
                var = target.id
            elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                var = target.attr
            else:
                var = None
            if var and var in (EQ_VARS['feedforward'] | EQ_VARS['backpropagation']):
                seg = ast.get_source_segment(src, stmt.value)
                if seg is None:
                    seg = ast.unparse(stmt.value) if hasattr(ast, 'unparse') else ''
                line = norm_expr(f"{var} = {seg}")
                # skip trivial storage assignments like Z = Z, DX = DX
                if re.match(rf"^\s*{re.escape(var)}\s*=\s*{re.escape(var)}\s*$", line):
                    continue
                ordered.append((var, line))
    return ordered


def find_method_def(class_node: ast.ClassDef, method_name: str) -> ast.FunctionDef | None:
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    return None


def build_class_map(tree: ast.Module) -> Dict[str, ast.ClassDef]:
    return {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}


def get_base_names(cls: ast.ClassDef) -> List[str]:
    names = []
    for b in cls.bases:
        if isinstance(b, ast.Name):
            names.append(b.id)
        elif isinstance(b, ast.Attribute):
            names.append(b.attr)
    return names


def resolve_method_equations(src: str, class_map: Dict[str, ast.ClassDef], cls_name: str, method_name: str) -> List[Tuple[str, str]]:
    cls = class_map.get(cls_name)
    if not cls:
        return []
    # If method is defined in this class, extract
    method = find_method_def(cls, method_name)
    eqs = extract_method_equations_from_def(src, method) if method else []
    # Special case: if SReLULayer backpropagate, include ActivationLayer backprop equations first
    if cls_name == 'SReLULayer' and method_name == 'backpropagate':
        eqs = resolve_method_equations(src, class_map, 'ActivationLayer', 'backpropagate') + eqs
        return eqs
    # If not defined (or empty), look into bases
    if not eqs:
        for base in get_base_names(cls):
            if base in class_map:
                eqs = resolve_method_equations(src, class_map, base, method_name)
                if eqs:
                    break
    return eqs


def extract_layer_equations(layers_file: str) -> Dict[str, Dict[str, List[str]]]:
    with open(layers_file, 'r', encoding='utf-8') as f:
        src = f.read()
    tree = ast.parse(src)
    class_map = build_class_map(tree)

    layers: Dict[str, Dict[str, List[str]]] = {}
    for cls_name in LAYER_KEYS:
        node = class_map.get(cls_name)
        if not node:
            continue
        feed = resolve_method_equations(src, class_map, cls_name, 'feedforward')
        back = resolve_method_equations(src, class_map, cls_name, 'backpropagate')
        feed_eqs = [eq for var, eq in feed if var in EQ_VARS['feedforward']]
        back_eqs = [eq for var, eq in back if var in EQ_VARS['backpropagation']]
        layers[cls_name] = {
            'feedforward': feed_eqs,
            'backpropagation': back_eqs,
        }
    return layers


def extract_test_sections_from_file(path: str) -> List[Dict[str, List[str]]]:
    """Parse test file and extract equation blocks under '# feedforward' and '# backpropagation'.
    Returns a list of blocks, each block is a dict with keys 'feedforward' and 'backpropagation' mapping to lists of normalized assignment lines.
    """
    blocks: List[Dict[str, List[str]]] = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    n = len(lines)
    block_idx = 0
    while i < n:
        if lines[i].strip().startswith('# feedforward'):
            block = {'feedforward': [], 'backpropagation': [], '__meta__': {'file': path, 'index': block_idx}}
            block_idx += 1
            i += 1
            # collect feedforward assignments until blank line or next comment starting with '#'
            while i < n and lines[i].strip() and not lines[i].lstrip().startswith('#'):
                line = lines[i]
                if '=' in line:
                    block['feedforward'].append(norm_expr(line))
                i += 1
            # advance to backpropagation
            while i < n and not lines[i].strip().startswith('# backpropagation'):
                i += 1
            if i < n and lines[i].strip().startswith('# backpropagation'):
                i += 1
                while i < n and lines[i].strip() and not lines[i].lstrip().startswith('#'):
                    line = lines[i]
                    if '=' in line:
                        block['backpropagation'].append(norm_expr(line))
                    i += 1
            blocks.append(block)
        else:
            i += 1
    return blocks


def extract_all_test_sections(tests_dir: str) -> List[Dict[str, List[str]]]:
    all_blocks: List[Dict[str, List[str]]] = []
    for fname in os.listdir(tests_dir):
        if fname.startswith('test_layer_') and fname.endswith('.py'):
            path = os.path.join(tests_dir, fname)
            blocks = extract_test_sections_from_file(path)
            # Normalize left-hand variable names to filter only relevant ones and maintain order
            norm_blocks: List[Dict[str, List[str]]] = []
            for block in blocks:
                new_block = {'feedforward': [], 'backpropagation': [], '__meta__': block.get('__meta__', {'file': path, 'index': 0})}
                for section in ('feedforward', 'backpropagation'):
                    for line in block[section]:
                        m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
                        if not m:
                            continue
                        var = m.group(1)
                        if var in EQ_VARS[section]:
                            new_block[section].append(line)
                norm_blocks.append(new_block)
            all_blocks.extend(norm_blocks)
    return all_blocks


def match_equations(layer_eqs: Dict[str, List[str]], block: Dict[str, List[str]]) -> bool:
    # Compare equations ignoring order but requiring exact text match after normalization
    return (sorted(layer_eqs['feedforward']) == sorted(block['feedforward']) and
            sorted(layer_eqs['backpropagation']) == sorted(block['backpropagation']))


def block_distance(layer_eqs: Dict[str, List[str]], block: Dict[str, List[str]]) -> int:
    # Distance as total symmetric difference in equations between layer and block
    dist = 0
    for section in ('feedforward', 'backpropagation'):
        a = set(layer_eqs.get(section, []))
        b = set(block.get(section, []))
        dist += len(a.symmetric_difference(b))
    return dist


def print_block_equations(prefix: str, block: Dict[str, List[str]]) -> None:
    meta = block.get('__meta__', {})
    src = meta.get('file', '<unknown>')
    idx = meta.get('index', '?')
    print(f"{prefix} (from {src}, block #{idx}) feedforward:")
    for line in block.get('feedforward', []):
        print('  ' + line)
    print(f"{prefix} (from {src}, block #{idx}) backpropagation:")
    for line in block.get('backpropagation', []):
        print('  ' + line)


def setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format='[%(levelname)s] %(message)s')


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Check consistency of layer equations with SymPy tests')
    parser.add_argument('-v', '--verbose', action='count', default=0, help='Increase verbosity (use -vv for debug)')
    parser.add_argument('--layers-file', default=LAYERS_FILE_DEFAULT, help='Path to layers.py to inspect (default: src/nerva_jax/layers.py)')
    parser.add_argument('--sympy-repo', default=SYMPY_REPO_DEFAULT, help='Path to nerva-sympy repository (default: ../nerva-sympy)')
    parser.add_argument('--tests-subdir', default='tests', help='Relative path to tests dir inside nerva-sympy (default: tests)')
    args = parser.parse_args(argv)

    setup_logging(args.verbose)

    tests_dir = os.path.join(args.sympy_repo, args.tests_subdir)
    layers = extract_layer_equations(args.layers_file)
    blocks = extract_all_test_sections(tests_dir)

    logging.info(f'Found {len(layers)} layers with equations to check.')
    logging.info(f'Found {len(blocks)} test equation blocks across files.')

    any_mismatch = False
    for layer_name, eqs in layers.items():
        # Choose best matching SymPy block (minimal distance)
        best_block = min(blocks, key=lambda b: block_distance(eqs, b)) if blocks else None

        def section_matches(section: str) -> bool:
            if not best_block:
                return False
            return sorted(eqs.get(section, [])) == sorted(best_block.get(section, []))

        ff_ok = section_matches('feedforward')
        bp_ok = section_matches('backpropagation')

        if ff_ok and bp_ok:
            meta = best_block.get('__meta__', {}) if best_block else {}
            logging.info(f'Layer {layer_name}: equations match (from {meta.get("file")} block #{meta.get("index")})')
            if args.verbose:
                # Show both sides in verbose mode even if matching
                print(f'[{layer_name}] equations from layers.py:')
                print('  feedforward:')
                for line in eqs.get('feedforward', []):
                    print('    ' + line)
                print('  backpropagation:')
                for line in eqs.get('backpropagation', []):
                    print('    ' + line)
                if best_block:
                    print_block_equations(f'[{layer_name}] matching SymPy test', best_block)
            continue

        # At least one section mismatched
        any_mismatch = True
        meta = best_block.get('__meta__', {}) if best_block else {}
        logging.warning(f'Layer {layer_name}: mismatch against SymPy (closest {meta.get("file")} block #{meta.get("index")})')

        def print_section(title: str, ours: List[str], theirs: List[str]):
            print(f'[{layer_name}] {title}:')
            print('  ours:')
            for line in ours:
                print('    ' + line)
            if best_block:
                print('  sympy:')
                for line in theirs:
                    print('    ' + line)

        if args.verbose:
            # In verbose mode, show both sections for full context
            print_section('feedforward', eqs.get('feedforward', []), best_block.get('feedforward', []) if best_block else [])
            print_section('backpropagation', eqs.get('backpropagation', []), best_block.get('backpropagation', []) if best_block else [])
        else:
            # In normal mode, only print mismatched sections
            if not ff_ok:
                print_section('feedforward', eqs.get('feedforward', []), best_block.get('feedforward', []) if best_block else [])
            if not bp_ok:
                print_section('backpropagation', eqs.get('backpropagation', []), best_block.get('backpropagation', []) if best_block else [])

    if any_mismatch:
        return 1
    else:
        print('Success: Matching test equations were found for all layers.')
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
