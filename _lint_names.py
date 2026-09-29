"""简易未定义全局名检查：找出文件里用到但没 import / 没定义的名字。

比 pyflakes 轻，够用：只报"疑似 NameError 风险"的全局名（np / cv2 / os 等）。
"""
import ast
import builtins
import os
import sys

BUILTIN = set(dir(builtins))
# 各模块自己定义的全局（函数/类/变量）在下面收集
SKIP_DIRS = {"__pycache__", "build", "dist", "backup_tt", "_internal"}


def collect_defined(tree):
    """收集模块顶层定义名 + import 绑定名 + 赋值目标 + 函数内赋值（粗）"""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                names.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                names.add(a.asname or a.name)
        elif isinstance(node, ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
            # 参数也算本地
            args = node.args
            for a in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs):
                names.add(a.arg)
            if args.vararg:
                names.add(args.vararg.arg)
            if args.kwarg:
                names.add(args.kwarg.arg)
        elif isinstance(node, ast.Name):
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                names.add(node.id)
        elif isinstance(node, ast.ExceptHandler):
            if node.name:
                names.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
        elif isinstance(node, ast.comprehension):
            # 推导式目标
            for n in ast.walk(node.target):
                if isinstance(n, ast.Name):
                    names.add(n.id)
        elif isinstance(node, ast.Lambda):
            args = node.args
            for a in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs):
                names.add(a.arg)
    return names


def used_globals(tree):
    """被 Load 引用、且不是某个函数局部的名字（粗：全部 Load 名）"""
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            out.setdefault(node.id, node.lineno)
    return out


def scan(path):
    try:
        src = open(path, encoding="utf-8", errors="replace").read()
        tree = ast.parse(src, filename=path)
    except SyntaxError as e:
        return [f"  语法错误 line{e.lineno}: {e.msg}"]
    defined = collect_defined(tree)
    issues = []
    for name, lineno in sorted(used_globals(tree).items(), key=lambda x: x[1]):
        if name in BUILTIN or name in defined:
            continue
        # 常见误报：__name__ / __file__ 等
        if name.startswith("__") and name.endswith("__"):
            continue
        issues.append(f"  line {lineno}: 未定义名字 '{name}'")
    return issues


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    for fn in sorted(os.listdir(root)):
        if not fn.endswith(".py") or fn.startswith("_"):
            continue
        p = os.path.join(root, fn)
        res = scan(p)
        if res:
            print(f"\n== {fn} ==")
            for r in res:
                print(r)
    print("\n扫描完成")


if __name__ == "__main__":
    main()
