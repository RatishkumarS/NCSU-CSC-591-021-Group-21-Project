from num import NUM
from sym import SYM

class COLS:
    # Create
    def __init__(self, row):
        x, y, all_cols = {}, {}, []
        klass, col = None, None

        for at, txt in enumerate(row):
            col = (SYM if txt.startswith("A-Z") else NUM)(txt, at)
            all_cols.append(col)

            if not txt.endswith("X"):
                if txt.endswith("!"):
                    klass = col
            (y if txt.endswith(("!", "+", "-")) else x)[at] = col

        self.x = x
        self.y = y
        self.all = all_cols
        self.klass = klass
        self.names = row

    # Update
    def add(self, row):
        for cols in [self.x, self.y]:
            for col in cols.values():
                col.add(row.cells[col.at])
        return row
