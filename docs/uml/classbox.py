#!/usr/bin/env python3
"""classbox.py - expand compact UML class boxes in a Graphviz file.

The class diagrams (docs/uml/*.gv) are ordinary Graphviz files, except that a
node whose label starts with the word UML is written as plain text:

    Creature [label="UML
      Creature
      --
      + id: String
      + gen: Int
      --
      + operation(x: Int): Void
    "];

Lines are: an optional stereotype line starting with « (or {abstract}),
the class name, then compartments separated by lines holding only "--"
(attributes first, then operations). A line "==" instead of "--" starts
the operations with the attribute compartment suppressed.
This script turns each such label into an HTML-like table with true UML
compartments (bold centred name, horizontal rules between compartments,
left-aligned members) and writes the result to standard output.

A package frame is written  label="PKG Engine 400x300"  (name, then the
size of the frame body in points).

Every member row gets a port, so that edges can attach at a chosen height:
  Creature:t      the name compartment
  Creature:a3     the third attribute row     (use e.g. Creature:a3:w)
  Creature:o2     the second operation row
Usage:  python3 classbox.py uml_datamodel.gv > uml_datamodel.dot
"""
import html
import re
import sys

LABEL = re.compile(r'label="UML\n((?:\\.|[^"\\])*)"', re.S)
PKG = re.compile(r'label="PKG ([^"]*?) (\d+)x(\d+)"')


def package(m):
    """A UML package frame: a name tab on top of a large empty box.

    Written as  label="PKG <name> <width>x<height>"  (points). Package
    nodes are declared first so that they are drawn behind the classes.
    """
    name, w, h = m.group(1), int(m.group(2)), int(m.group(3))
    return ('label=<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">'
            f'<TR><TD BORDER="1" SIDES="LTR" CELLPADDING="3" ALIGN="LEFT"><B>{esc(name)}</B></TD>'
            '<TD></TD></TR>'
            f'<TR><TD COLSPAN="2" BORDER="1" WIDTH="{w}" HEIGHT="{h}" FIXEDSIZE="TRUE"></TD></TR>'
            '</TABLE>>')


def esc(text):
    return html.escape(text, quote=False)


def box(body):
    body = body.replace('\\"', '"')
    lines = [l.strip() for l in body.strip("\n").split("\n")]
    lines = [l for l in lines if l != ""]
    head = []
    while lines and lines[0] not in ("--", "=="):
        head.append(lines.pop(0))
    parts, cur = [], None
    for l in lines:
        if l == "--":
            cur = []
            parts.append(cur)
        elif l == "==":
            # Operations only: the attribute compartment is suppressed.
            cur = []
            parts.append(None)
            parts.append(cur)
        else:
            cur.append(l)
    stereo = [h for h in head[:-1]]
    name = head[-1]
    abstract = any(s == "{abstract}" for s in stereo)
    rows = []
    title = "".join(f"<I>{esc(s)}</I><BR/>" for s in stereo)
    title += f"<B><I>{esc(name)}</I></B>" if abstract else f"<B>{esc(name)}</B>"
    rows.append(f'<TR><TD PORT="t" CELLPADDING="4">{title}</TD></TR>')
    for k, part in enumerate(parts):
        prefix = "a" if k == 0 else "o"
        if part is None:
            continue
        rows.append("<HR/>")
        if not part:
            rows.append('<TR><TD CELLPADDING="3"> </TD></TR>')
            continue
        for i, member in enumerate(part, 1):
            # Thin spacer rows above the first and below the last member.
            if i == 1:
                rows.append('<TR><TD CELLPADDING="0" HEIGHT="3" FIXEDSIZE="FALSE"></TD></TR>')
            text = esc(member)
            # {static} members are underlined, as in UML.
            if text.startswith("_") and text.endswith("_"):
                text = f"<U>{text[1:-1]}</U>"
            rows.append(f'<TR><TD ALIGN="LEFT" PORT="{prefix}{i}">&#160;&#160;{text}&#160;&#160;</TD></TR>')
            if i == len(part):
                rows.append('<TR><TD CELLPADDING="0" HEIGHT="3" FIXEDSIZE="FALSE"></TD></TR>')
    return ('label=<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="1">'
            + "".join(rows) + "</TABLE>>")


def main(path):
    src = open(path, encoding="utf-8").read()
    src = PKG.sub(package, src)
    sys.stdout.write(LABEL.sub(lambda m: box(m.group(1)), src))


if __name__ == "__main__":
    main(sys.argv[1])
