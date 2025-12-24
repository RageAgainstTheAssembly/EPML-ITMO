from pathlib import Path

import mkdocs_gen_files

PACKAGE = "wine_predictor"
SRC_DIR = Path(PACKAGE)

nav = mkdocs_gen_files.Nav()

for path in sorted(SRC_DIR.rglob("*.py")):
    if path.name == "__init__.py":
        continue

    module_path = path.with_suffix("")
    parts = list(module_path.parts)
    module = ".".join(parts)

    full_doc_path = Path("reference", *parts).with_suffix(".md")

    doc_path_for_nav = Path(*parts).with_suffix(".md")
    nav[parts] = doc_path_for_nav.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as f:
        f.write(f"# `{module}`\n\n")
        f.write(f"::: {module}\n")

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
