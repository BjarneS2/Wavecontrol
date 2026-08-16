import random

from HCA import HCA, HCA1D, plot_move_grid, plot_slider




# # ---- 1D variant ----
# rng = random.Random(0)
# row = [1, 0, 1, 1, 0, 0, 1, 1, 1]
# print("".join("#" if x else "." for x in row))
# s = HCA1D(row, "best")
# s.solve(allow_outside=False, validate=True)
# print("K =", s.K, " moves =", len(s.moves))
# plot_slider(s)


if __name__ == "__main__":
    N, M, K = 8, 8, 4
    rng = random.Random(0)
    g = [[1 if rng.random() < 0.55 else 0 for _ in range(M)] for _ in range(N)]

    print("initial array (#=atom, .=empty):")
    for row in g:
        print("".join("#" if x else "." for x in row))

    # default mode: identical output to the reference HCA.py
    s = HCA(g, K, corner="best")
    s.solve(allow_outside=False, validate=True)

    # incremental_dist=True: faster, still valid, move sequence may differ
    s_fast = HCA(g, K, corner="best", incremental_dist=True)
    s_fast.solve(allow_outside=False, validate=True)

    print("\nfinal state (#=atom, [.]=target site):\n")
    s.print_state()
    print("\ntarget origin           :", s.target_origin)
    print("default      -> moves   :", len(s.moves))
    print("incremental  -> moves   :", len(s_fast.moves))

    print("\nexample roads (default; 'OUT' = looped around the array):")
    for mv, p in list(zip(s.moves, s.paths))[:4]:
        print("  %s -> %s   via %s" % (mv[0], mv[1], p))

    plot_slider(s)
