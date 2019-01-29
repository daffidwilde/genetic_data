""" Tell pytest not to collect tests from reST orphan files. """

import os

collect_ignore = []
for root, dirs, files in os.walk("./docs/"):
    for f in files:
        if f.endswith(".rst"):
            f = os.path.join(root, f)
            with open(f, "r") as rst:
                text = str(rst.read())
                if text.startswith(".. :orphan:"):
                    print(f)
                    collect_ignore.append(f)
