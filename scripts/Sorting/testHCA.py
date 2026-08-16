import random

from HCA import HCA, HCA1D, plot_move_grid, plot_slider
import time

N, M, K = 8, 8, 4
rng = random.Random(0)
g = [[1 if rng.random() < 0.5 else 0 for _ in range(M)] for _ in range(N)]

print("initial array (#=atom, .=empty):")
for row in g:
    print("".join("#" if x else "." for x in row))

a = time.time()
s_out = HCA(g, K, corner="best", incremental_dist=True)
s_out.solve(allow_outside=False, validate=True)
b= time.time()
s_in = HCA(g, K, corner="best")
s_in.solve(allow_outside=False, validate=True)
c= time.time()
print("TIME (allow_outside=True) :", b-a)
print("TIME (allow_outside=False):", c-b)
print("")

print("\nfinal state (#=atom, [.]=target site):\n")
s_out.print_state()
print("\ntarget origin       :", s_out.target_origin)
print("allow_outside=True  -> moves:", len(s_out.moves))
print("allow_outside=False -> moves:", len(s_in.moves))

print("\nexample roads (allow_outside=True; 'OUT' = looped around the array):")
for mv, p in list(zip(s_out.moves, s_out.paths))[:4]:
    print("  %s -> %s   via %s" % (mv[0], mv[1], p))

plot_move_grid(s_out, include_initial=True)
plot_slider(s_in)


# if __name__ == "__main__":
#     rng = random.Random(0)
#     row = [1, 0, 1, 1, 0, 0, 1, 1, 1]# [1 if rng.random() < 0.5 else 0 for _ in range(10)]
#     print("initial 1D array (#=atom, .=empty):")
#     print("".join("#" if x else "." for x in row))

#     s = HCA1D(row, "best")
#     s.solve(allow_outside=False, validate=True)
#     print("K =", s.K, " moves =", len(s.moves))

#     plot_slider(s)