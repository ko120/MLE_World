"""Public-schema and syntax evidence for validation review; never executes candidates.

Ported from the September 29 offline candidate. These bounded recognizers are
not a general Python interpreter. They supply evidence for model review;
unsupported forms yield no mechanical evidence, not a correctness verdict.
"""

import ast
import copy
import io
import itertools
import tokenize


def unreachable_function_lines(code):
    tree = ast.parse(code)
    # Reflection/dynamic evaluation can reach names absent from syntactic loads.
    if any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id in {"eval", "exec", "globals", "locals", "getattr", "__import__"}
        for n in ast.walk(tree)
    ):
        return set()
    definitions = {
        n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    def references(nodes):
        return {
            n.id
            for node in nodes
            for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }

    live = (
        references(
            [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        )
        & definitions.keys()
    )
    pending = list(live)
    while pending:
        name = pending.pop()
        for ref in references([definitions[name]]) & definitions.keys() - live:
            live.add(ref)
            pending.append(ref)
    # Decorators execute at definition time and can register a function.
    if any(n.decorator_list for n in definitions.values()):
        return set()
    return {
        line
        for name, node in definitions.items()
        if name not in live
        for line in range(node.lineno, node.end_lineno + 1)
    }


def schema_branch_facts(code, public_schema):
    train, test = train_test_pair(public_schema)
    if "key_presence_counts" not in train or "key_presence_counts" not in test:
        return []
    train_cols = {k for k, v in train["key_presence_counts"].items() if v}
    test_cols = {k for k, v in test["key_presence_counts"].items() if v}
    raw = train_cols | test_cols
    tree = ast.parse(code)
    constants = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            if isinstance(value, (str, int, float, list, tuple)):
                constants[node.targets[0].id] = value

    def resolve(node, env, cols):
        if isinstance(node, ast.Name):
            return env[node.id]
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.List, ast.Tuple)):
            return [resolve(n, env, cols) for n in node.elts]
        if isinstance(node, ast.Attribute) and node.attr == "columns":
            return cols
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left, right = resolve(node.left, env, cols), resolve(node.comparators[0], env, cols)
            if isinstance(node.ops[0], ast.In):
                return left in right
            if isinstance(node.ops[0], ast.NotIn):
                return left not in right
        if isinstance(node, ast.BoolOp):
            values = [resolve(n, env, cols) for n in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"all", "any"}
            and len(node.args) == 1
            and isinstance(node.args[0], ast.GeneratorExp)
        ):
            g = node.args[0]
            if (
                len(g.generators) == 1
                and isinstance(g.generators[0].target, ast.Name)
                and not g.generators[0].ifs
            ):
                values = [
                    resolve(g.elt, {**env, g.generators[0].target.id: item}, cols)
                    for item in resolve(g.generators[0].iter, env, cols)
                ]
                return all(values) if node.func.id == "all" else any(values)
        raise ValueError("Unsupported expression")

    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    facts = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not any(
            isinstance(n, ast.Attribute) and n.attr == "columns" for n in ast.walk(node.test)
        ):
            continue
        # Substituting the SAME schema for train.columns AND test.columns
        # invents reachability for an explicit intersection guard. This older
        # counterfactual evaluator only supports a single dataframe expression.
        frames = {
            ast.unparse(n.value) for n in ast.walk(node.test)
            if isinstance(n, ast.Attribute) and n.attr == "columns"
        }
        if len(frames) != 1:
            continue
        bindings = [dict(constants)]
        p = parents.get(node)
        while p:
            if isinstance(p, ast.For) and isinstance(p.target, ast.Name):
                try:
                    iterable = resolve(p.iter, constants, train_cols)
                    if not isinstance(iterable, (list, tuple)) or len(iterable) > 100:
                        break
                    bindings = [
                        {**env, p.target.id: v} for env, v in itertools.product(bindings, iterable)
                    ]
                except (KeyError, ValueError, TypeError):
                    break
            p = parents.get(p)
        for env in bindings:
            try:
                t, u = resolve(node.test, env, train_cols), resolve(node.test, env, test_cols)
                if not isinstance(t, bool) or not isinstance(u, bool):
                    continue
                if t == u:
                    continue
                referenced = sorted(
                    {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)} & env.keys()
                )
                refs = {name: env[name] for name in referenced}
                fields = set()
                for value in refs.values():
                    if isinstance(value, str):
                        fields.add(value)
                    elif isinstance(value, (list, tuple)):
                        fields.update(x for x in value if isinstance(x, str))
                fields.update(
                    n.value
                    for n in ast.walk(node.test)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)
                )
                if not fields or not fields <= raw:
                    continue
                facts.append(
                    {
                        "condition_line": node.lineno,
                        "condition": ast.unparse(node.test),
                        "bindings": refs,
                        "if_evaluated_on_raw_labeled_columns": t,
                        "if_evaluated_on_raw_test_columns": u,
                        "then_lines": [node.body[0].lineno, node.body[-1].end_lineno],
                        "else_lines": (
                            [node.orelse[0].lineno, node.orelse[-1].end_lineno]
                            if node.orelse
                            else []
                        ),
                        "limit": "Mechanical membership on original input schemas. Verify this dataframe still has those columns and that the branch output reaches prediction; this table is not a defect verdict.",
                    }
                )
            except (KeyError, ValueError, TypeError):
                continue
    return facts


def _source_text(code, stripped):
    if not stripped:
        return "\n".join(f"L{i}: {s}" for i, s in enumerate(code.splitlines(), 1))
    # Omit comments/docstrings without renumbering executable lines.
    doclines = unreachable_function_lines(code)
    for node in ast.walk(ast.parse(code)):
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            doclines.update(range(body[0].lineno, body[0].end_lineno + 1))
    comments = {
        t.start[0]: t.start[1]
        for t in tokenize.generate_tokens(io.StringIO(code).readline)
        if t.type == tokenize.COMMENT
    }
    lines = []
    for i, line in enumerate(code.splitlines(), 1):
        if i in doclines:
            continue
        line = line[: comments[i]] if i in comments else line
        if line.strip():
            lines.append(f"L{i}: {line}")
    return "\n".join(lines)


def compact_schema(schema):
    train, test = train_test_pair(schema)
    if "key_presence_counts" in train and "key_presence_counts" in test:
        both = set(train["key_presence_counts"]) | set(test["key_presence_counts"])
        return {
            "raw_fields": {
                k: {
                    "train": train["key_presence_counts"].get(k, 0),
                    "test": test["key_presence_counts"].get(k, 0),
                }
                for k in sorted(both)
            },
            "train_rows": train["rows"],
            "test_rows": test["rows"],
            "absent_from_all_test_rows": sorted(
                set(train["key_presence_counts"]) - set(test["key_presence_counts"])
            ),
        }
    result = copy.deepcopy(schema)
    result["column_dtypes"] = input_column_dtypes(schema)
    for record in result.get("files", []):
        name = record.get("file", "").lower().replace("_", "")
        if record.get("role") == "sample" or "samplesubmission" in name:
            record.pop("dtypes", None)
            record.pop("column_dtypes", None)
            # Older cached schemas guessed that column zero was the ID. Do not
            # retain that guess when it is actually a prediction column.
            first = (record.get("columns") or [""])[0]
            normalized = "".join(c.lower() for c in first if c.isalnum())
            if not record.get("id_column") and normalized not in {
                "id", "image", "filename", "rowid", "imageid", "key",
            }:
                record.pop("id_examples", None)
            record["semantics"] = "Example template; not a required dtype, column order, or ID encoding."
    return result


def input_column_dtypes(schema):
    """Never let placeholder predictions supply raw-input selector evidence."""
    train, test = train_test_pair(schema)
    names = {f.get("file") for f in (train, test)}
    return {k: v for k, v in schema.get("column_dtypes", {}).items() if k in names}


def column_filter_facts(code, schema):
    """Show potential exclusion operations without pretending to execute Python.

    Intersections after derived features or padding are NOT proof of equivalent
    information. The verifier must trace the actual frame at each cited line.
    """
    train, test = train_test_pair(schema)
    train_fields = set(train.get("columns", train.get("key_presence_counts", {})))
    test_fields = set(test.get("columns", test.get("key_presence_counts", {})))
    operations = []
    for node in ast.walk(ast.parse(code)):
        intersection = isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd)
        intersection |= (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                         and node.func.attr == "intersection")
        membership = isinstance(node, (ast.If, ast.ListComp, ast.SetComp, ast.GeneratorExp)) and any(
            isinstance(n, ast.Attribute) and n.attr == "columns" for n in ast.walk(node.test if isinstance(node, ast.If) else node)
        )
        if intersection or membership:
            expr = node.test if isinstance(node, ast.If) else node
            operations.append({"line": node.lineno, "expression": ast.unparse(expr)[:1000]})
    return {
        "shared_raw_columns": sorted(train_fields & test_fields),
        "train_only_raw_columns": sorted(train_fields - test_fields),
        "possible_column_filters": operations[:64],
        "limit": "Syntax evidence only. Trace named frames, intervening derived features and mutations before using these filters as exclusion evidence.",
    }


def field_literal_lines(code, field):
    """Exact field references only; similar names and comments are not evidence."""
    omitted = unreachable_function_lines(code)
    return {
        n.lineno for n in ast.walk(ast.parse(code))
        if isinstance(n, ast.Constant) and n.value == field and n.lineno not in omitted
    }


def project(code, stripped=True):
    if not stripped:
        return _source_text(code, False)
    tree = ast.parse(code)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    # Name introspection, evaluation or decorators can hide calls; preserve source.
    if any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id in {"getattr", "eval", "exec", "globals", "locals", "__import__"}
        for n in ast.walk(tree)
    ):
        return _source_text(code, True)
    omitted = set()
    dedent = {}
    for fn in tree.body:
        if (
            not isinstance(fn, ast.FunctionDef)
            or fn.decorator_list
            or fn.args.vararg
            or fn.args.kwarg
        ):
            continue
        if sum(isinstance(n, ast.FunctionDef) and n.name == fn.name for n in tree.body) != 1 or any(
            isinstance(n, ast.Name) and n.id == fn.name and isinstance(n.ctx, ast.Store)
            for n in ast.walk(tree)
        ):
            continue
        refs = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Name) and n.id == fn.name and isinstance(n.ctx, ast.Load)
        ]
        if not refs or any(
            not isinstance(parents.get(n), ast.Call) or parents[n].func is not n for n in refs
        ):
            continue
        calls = [parents[n] for n in refs]
        names = [a.arg for a in fn.args.posonlyargs + fn.args.args]
        defaults = dict(zip(names[len(names) - len(fn.args.defaults) :], fn.args.defaults))
        defaults.update(
            {a.arg: d for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults) if d is not None}
        )
        bindings = []
        valid = True
        for call in calls:
            if any(isinstance(a, ast.Starred) for a in call.args) or any(
                k.arg is None for k in call.keywords
            ):
                valid = False
                break
            bound = {
                **defaults,
                **dict(zip(names, call.args)),
                **{k.arg: k.value for k in call.keywords},
            }
            bindings.append(bound)
        if not valid:
            continue
        constants = {}
        for name in set.intersection(*(set(b) for b in bindings)):
            vals = [b[name] for b in bindings]
            if (
                all(isinstance(v, ast.Constant) and type(v.value) is bool for v in vals)
                and len({v.value for v in vals}) == 1
            ):
                # Do not substitute a parameter reassigned inside its function.
                if not any(
                    isinstance(n, ast.Name)
                    and n.id == name
                    and isinstance(n.ctx, (ast.Store, ast.Del))
                    for n in ast.walk(fn)
                ):
                    constants[name] = vals[0].value
        for n in ast.walk(fn):
            if not isinstance(n, ast.If):
                continue
            parent = parents[n]
            while (
                parent is not fn
                and parent is not None
                and not isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            ):
                parent = parents.get(parent)
            if parent is not fn:
                continue
            test = n.test
            negate = False
            if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
                test = test.operand
                negate = True
            if not isinstance(test, ast.Name) or test.id not in constants:
                continue
            value = constants[test.id] ^ negate
            chosen = n.body if value else n.orelse
            selected = {line for c in chosen for line in range(c.lineno, c.end_lineno + 1)}
            omitted.update(set(range(n.lineno, n.end_lineno + 1)) - selected)
            if chosen:
                shift = chosen[0].col_offset - n.col_offset
                for line in selected:
                    dedent[line] = dedent.get(line, 0) + shift
    rendered = []
    for line in _source_text(code, True).splitlines():
        prefix, raw = line.split(": ", 1)
        number = int(prefix[1:])
        if number in omitted:
            continue
        shift = dedent.get(number, 0)
        rendered.append(prefix + ": " + raw[shift:])
    return "\n".join(rendered)


def numeric_selection_facts(code, schema):
    dtypes = input_column_dtypes(schema)
    if not dtypes:
        return []
    tree = ast.parse(code)
    constants = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            try:
                v = ast.literal_eval(n.value)
            except (ValueError, TypeError):
                continue
            constants[n.targets[0].id] = v

    def evaluate(n, col, dtype, symbol):
        if isinstance(n, ast.Name):
            return col if n.id == symbol else constants[n.id]
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, (ast.List, ast.Tuple, ast.Set)):
            return [evaluate(x, col, dtype, symbol) for x in n.elts]
        if (
            isinstance(n, ast.Attribute)
            and n.attr == "dtype"
            and isinstance(n.value, ast.Subscript)
            and isinstance(n.value.slice, ast.Name)
            and n.value.slice.id == symbol
        ):
            return dtype
        if isinstance(n, ast.BoolOp):
            vv = [evaluate(x, col, dtype, symbol) for x in n.values]
            return all(vv) if isinstance(n.op, ast.And) else any(vv)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            return not evaluate(n.operand, col, dtype, symbol)
        if isinstance(n, ast.Compare) and len(n.ops) == 1:
            a, b = evaluate(n.left, col, dtype, symbol), evaluate(
                n.comparators[0], col, dtype, symbol
            )
            if isinstance(n.ops[0], ast.In):
                return a in b
            if isinstance(n.ops[0], ast.NotIn):
                return a not in b
            if isinstance(n.ops[0], ast.Eq):
                return a == b
            if isinstance(n.ops[0], ast.NotEq):
                return a != b
        raise ValueError("Unsupported selector")

    result = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.ListComp) or len(n.generators) != 1:
            continue
        g = n.generators[0]
        if (
            not isinstance(g.target, ast.Name)
            or not isinstance(g.iter, ast.Attribute)
            or g.iter.attr != "columns"
        ):
            continue
        if not any(
            isinstance(x, ast.Attribute) and x.attr == "dtype"
            for test in g.ifs
            for x in ast.walk(test)
        ):
            continue
        rows = {}
        try:
            for file, types in dtypes.items():
                rows[file] = [
                    col
                    for col, dtype in types.items()
                    if all(evaluate(test, col, dtype, g.target.id) for test in g.ifs)
                ]
        except (KeyError, ValueError, TypeError):
            continue
        result.append(
            {
                "line": n.lineno,
                "expression": ast.unparse(n),
                "raw_public_columns_satisfying_predicate": rows,
                "limit": "This evaluates the source selector on original public dtypes. It is not proof that all columns reach prediction; trace subsequent filters and consumers. Derived columns are not included in this raw-column list.",
            }
        )
    return result


def helpers(code):
    t = ast.parse(code)
    names = {n.name for n in t.body if isinstance(n, ast.FunctionDef)}
    return [
        {"line": n.lineno, "call": ast.unparse(n)}
        for n in ast.walk(t)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in names
    ]


def train_test_pair(schema):
    files = schema.get("files", [])

    def candidates(role):
        return [
            f
            for f in files
            if f.get("role") == role
            or (
                not f.get("role")
                and role in f.get("file", "").lower()
                and ("columns" in f or "key_presence_counts" in f)
            )
        ]

    train, test = candidates("train"), candidates("test")
    return (train[0], test[0]) if len(train) == len(test) == 1 else ({}, {})


def absent_fields(schema):
    train, test = train_test_pair(schema)
    targets = set(schema.get("designated_raw_target_fields", []))
    if "key_presence_counts" in train and "key_presence_counts" in test:
        return (
            {k for k, v in train["key_presence_counts"].items() if v}
            - {k for k, v in test["key_presence_counts"].items() if v}
            - targets
        )
    if "columns" in train and "columns" in test:
        return set(train["columns"]) - set(test["columns"]) - targets
    return set()
