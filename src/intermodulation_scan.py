"""intermodulation_scan.py
Power / distortion scan for a 2D multi-tone AOD grid.

    python intermodulation_scan.py --nx 8 --ny 8 --pin 200 --targets 0.05 0.1 0.2 0.5 1.0
    or --sigma-scan (for scanning instead of targets)

sigma is the total rms drive in radians; it is the single knob that sets
the power / distortion trade.  Per-spot power is NOT monotone in sigma
(it peaks near sigma ~ 1.2), so every target is solved on the low branch.
"""

import argparse

import numpy as np

from intermodulation_simulation import (
    line_spectrum,
    make_comb,
    make_weights,
    metrics,
    optimize_joint,
    optimize_phases,
    predistort,
    schroeder_phases,
    suggest_M,
)


def sigma_to_a(K, sigma):
    return np.full(K, sigma * np.sqrt(2.0 / K))


def flatness_vs_mean_dB(spot_powers):
    """Worst per-spot deviation from the mean spot power, in dB."""
    mean = np.mean(spot_powers)
    return float(np.max(np.abs(10 * np.log10(spot_powers / mean))))


def spot_power(nx, ny, sigma, optimize=False, **kw):
    """Per-spot power (fraction of P_in), worst ghost, worst
    flatness-vs-mean (dB), and IMD efficiency (total power landing in the
    spots / total input power) for a 2D grid."""
    res = []
    for n in (nx, ny):
        a = sigma_to_a(len(n), sigma)
        M = suggest_M(n, a)
        if optimize:
            a, phi = optimize_joint(n, a, M=M, **kw)
        else:
            phi = schroeder_phases(len(n))
        res.append((metrics(n, a, phi, M), a, phi))
    (mx, ax, px), (my, ay, py) = res
    P = float(np.mean(mx["spot_powers"]) * np.mean(my["spot_powers"]))
    flat = max(
        flatness_vs_mean_dB(mx["spot_powers"]), flatness_vs_mean_dB(my["spot_powers"])
    )
    # The 2D field factorizes, so total spot power = (sum of x-axis carrier
    # power) * (sum of y-axis carrier power) -- the fraction of P_in that
    # actually lands in a spot rather than being lost to IMD products.
    eff = float(mx["carrier_power"] * my["carrier_power"])
    return (
        P,
        max(mx["worst_ghost_dBc"], my["worst_ghost_dBc"]),
        flat,
        eff,
        (ax, px, ay, py),
    )


def predistort_metrics(nx, ny, sigma, B_avail=None, clip=1e-3, want_fields=False):
    """IMD-cancelling predistortion: drive with u=arcsin(target field) so
    sin(u) reproduces the target comb exactly, up to what B_avail (highest
    RF channel the chain passes) lets through -- see
    intermodulation_simulation.predistort. The target field uses Schroeder
    phases (kept low-crest-factor so it rarely needs the clip headroom).
    Same per-axis factorized 2D construction as spot_power. With
    B_avail=None (unlimited bandwidth) this is close to ideal/lossless --
    tighten B_avail to see a realistic RF chain degrade it."""
    res = []
    for n in (nx, ny):
        K = len(n)
        A_tgt = sigma_to_a(K, sigma)
        phi = schroeder_phases(K)
        _, _, P = predistort(n, A_tgt, phi, B_avail=B_avail, clip=clip)
        spots = np.asarray(n)
        is_spot = np.zeros(P.size, dtype=bool)
        is_spot[spots] = True
        ghost = np.where(is_spot, 0.0, P)
        ghost[0] = 0.0
        Pc = P[spots]
        ref = float(Pc.min())
        wg = 10 * np.log10(max(float(ghost.max()), 1e-300) / max(ref, 1e-300))
        res.append((Pc, wg, float(Pc.sum()), P if want_fields else None))
    (Pcx, gx, cx, Px), (Pcy, gy, cy, Py) = res
    P_mean = float(np.mean(Pcx) * np.mean(Pcy))
    flat = max(flatness_vs_mean_dB(Pcx), flatness_vs_mean_dB(Pcy))
    return P_mean, max(gx, gy), flat, cx * cy, (Px, Py)


def _max_carrier_power_amplitudes(n, phi, a0, M, flat_max_dB, bounds=(1e-4, np.pi)):
    """Given fixed phases, find the amplitude vector that maximizes total
    carrier power SUBJECT TO the same total drive budget as a0 (sum(a^2)
    fixed -- so this never secretly draws more RF power than the sigma it
    is being compared against) and a spot-to-spot spread capped at
    flat_max_dB. Perfect equalization (flatten_amplitudes' default) is just
    one feasible point in this constraint set (spread=0); this searches the
    whole budget-conserving, bounded-spread region for the best one."""
    from scipy.optimize import minimize

    spots = np.asarray(n)
    budget = float(np.sum(a0**2))

    def carrier_powers(a):
        return line_spectrum(n, a, phi, M)[1][spots]

    def neg_total(a):
        return -float(np.sum(carrier_powers(a)))

    def budget_eq(a):
        return float(np.sum(a**2)) - budget

    def flat_ineq(a):
        Pc = carrier_powers(a)
        dev_dB = 10 * np.log10(Pc / np.mean(Pc))
        return flat_max_dB - float(np.max(np.abs(dev_dB)))  # feasible if >= 0

    res = minimize(
        neg_total,
        a0,
        method="SLSQP",
        bounds=[bounds] * len(a0),
        constraints=[
            {"type": "eq", "fun": budget_eq},
            {"type": "ineq", "fun": flat_ineq},
        ],
        options={"maxiter": 150, "ftol": 1e-12},
    )
    return res.x if res.success else a0


def power_focused_metrics(
    nx, ny, sigma, ghost_margin_sites=3, flat_max_dB=1.0, n_starts=8, seed=0
):
    """Optimize phases against ONLY the ghosts within ghost_margin_sites
    channel-spacings of the array footprint -- farther spurs are given zero
    weight and the optimizer is free to ignore them entirely. Then, at the
    SAME total drive budget as sigma_to_a(K, sigma) implies (so this stays
    a fair comparison against schroeder/optimized at the same sigma), search
    for the amplitude distribution that maximizes total carrier power while
    keeping the spot-to-spot spread within flat_max_dB of the mean. Same
    per-axis factorized 2D construction as spot_power; returns an (a, phi)
    sol like spot_power, so fields_from_sol() works on it."""
    res = []
    for n in (nx, ny):
        K = len(n)
        a0 = sigma_to_a(K, sigma)
        M = suggest_M(n, a0)
        step = int(np.min(np.diff(np.sort(n)))) if K > 1 else 1
        w = make_weights(
            n, M, w_inside=1.0, w_outside=0.0, margin=ghost_margin_sites * step
        )
        phi, _ = optimize_phases(
            n,
            a0,
            M=M,
            w=w,
            lam_flat=1.0,
            n_starts=n_starts,
            seed=seed,
            select=lambda m: -m["carrier_power"],
        )
        a_best = _max_carrier_power_amplitudes(n, phi, a0, M, flat_max_dB)
        Pc = line_spectrum(n, a_best, phi, M)[1][n]
        # near_ghost_dBc with the SAME margin as the objective weighting --
        # the "worst ghost anywhere" (worst_ghost_dBc) will look worse than
        # the other strategies by design: it now includes the intentionally
        # -unweighted far ghosts (e.g. the outermost 3rd-order product,
        # which for a K-tone comb sits a full array-width beyond the edge).
        # near_ghost_dBc is the metric that actually reflects what this
        # strategy optimizes for.
        m = metrics(n, a_best, phi, M, margin=ghost_margin_sites * step)
        res.append(
            (
                Pc,
                m["worst_ghost_dBc"],
                m["near_ghost_dBc"],
                float(Pc.sum()),
                a_best,
                phi,
            )
        )
    (Pcx, gx, nx_, cx, ax_, phix), (Pcy, gy, ny_, cy, ay_, phiy) = res
    P_mean = float(np.mean(Pcx) * np.mean(Pcy))
    flat = max(flatness_vs_mean_dB(Pcx), flatness_vs_mean_dB(Pcy))
    return (
        P_mean,
        max(gx, gy),
        max(nx_, ny_),
        flat,
        cx * cy,
        (ax_, phix, ay_, phiy),
    )


def sigma_for_power(nx, ny, P_target, lo=0.05, hi=1.15, tol=1e-4, **kw):
    """Invert per-spot power -> sigma, bisecting on the low-drive branch."""
    Plo = spot_power(nx, ny, lo, **kw)[0]
    Phi = spot_power(nx, ny, hi, **kw)[0]
    if not Plo <= P_target <= Phi:
        raise ValueError("reachable range is [%.3e, %.3e]" % (Plo, Phi))
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if spot_power(nx, ny, mid, **kw)[0] < P_target else (lo, mid)
    return 0.5 * (lo + hi)


def hit_power(nx, ny, P_target, refine=2, **kw):
    """Bisect on Schroeder phases, then correct for what optimization shifts."""
    sg = sigma_for_power(nx, ny, P_target)
    P, ghost, flat, eff, sol = spot_power(nx, ny, sg, optimize=True, **kw)
    for _ in range(refine):
        sg *= float(np.sqrt(np.sqrt(P_target / max(P, 1e-30))))
        P, ghost, flat, eff, sol = spot_power(nx, ny, sg, optimize=True, **kw)
    return sg, P, ghost, flat, eff, sol


def find_sigma_max(nx, ny, lo=0.3, hi=2.5, n=400):
    """Locate the sigma that maximizes per-spot power (the 'ceiling').
    Coarse grid then a local refine, since spot_power(sigma) is smooth
    and single-peaked over this range."""
    sg = np.linspace(lo, hi, n)
    P = np.array([spot_power(nx, ny, s)[0] for s in sg])
    i = int(np.argmax(P))
    lo2, hi2 = sg[max(i - 1, 0)], sg[min(i + 1, n - 1)]
    sg2 = np.linspace(lo2, hi2, 50)
    P2 = np.array([spot_power(nx, ny, s)[0] for s in sg2])
    return float(sg2[np.argmax(P2)])


def sigma_range_scan(
    nx,
    ny,
    P_in_mW,
    sigma_lo,
    sigma_hi=None,
    n=10,
    margin=0.98,
    optimize=True,
    b_avail=None,
    pf_margin=3,
    pf_flat_max=1.0,
    pf_starts=8,
    **kw,
):
    """Sweep sigma in n steps from sigma_lo up to margin*sigma_max
    (or an explicit sigma_hi).  Returns rows like power_scan."""
    if sigma_hi is None:
        sigma_hi = margin * find_sigma_max(nx, ny)
    rows = []
    for sg in np.linspace(sigma_lo, sigma_hi, n):
        P, ghost, flat, eff, sol = spot_power(nx, ny, sg, optimize=optimize, **kw)
        Ps, ghosts, flats, effs, sol_s = spot_power(nx, ny, sg, optimize=False)
        Pp, ghostp, flatp, effp, _ = predistort_metrics(nx, ny, sg, B_avail=b_avail)
        Pw, ghostw, ghostw_roi, flatw, effw, solw = power_focused_metrics(
            nx,
            ny,
            sg,
            ghost_margin_sites=pf_margin,
            flat_max_dB=pf_flat_max,
            n_starts=pf_starts,
        )
        rows.append(
            (
                sg,
                P * P_in_mW,
                ghost,
                flat,
                eff,
                Ps * P_in_mW,
                ghosts,
                flats,
                effs,
                Pp * P_in_mW,
                ghostp,
                flatp,
                effp,
                Pw * P_in_mW,
                ghostw,
                ghostw_roi,
                flatw,
                effw,
                sol,
                sol_s,
                solw,
            )
        )
    return rows


def power_scan(
    nx,
    ny,
    P_in_mW,
    targets_mW,
    b_avail=None,
    pf_margin=3,
    pf_flat_max=1.0,
    pf_starts=8,
    **kw,
):
    rows = []
    for t in targets_mW:
        try:
            sg, P, ghost, flat, eff, sol = hit_power(nx, ny, t / P_in_mW, **kw)
            Ps, ghosts, flats, effs, sol_s = spot_power(nx, ny, sg, optimize=False)
            Pp, ghostp, flatp, effp, _ = predistort_metrics(nx, ny, sg, B_avail=b_avail)
            Pw, ghostw, ghostw_roi, flatw, effw, solw = power_focused_metrics(
                nx,
                ny,
                sg,
                ghost_margin_sites=pf_margin,
                flat_max_dB=pf_flat_max,
                n_starts=pf_starts,
            )
            rows.append(
                (
                    t,
                    sg,
                    P * P_in_mW,
                    ghost,
                    flat,
                    eff,
                    Ps * P_in_mW,
                    ghosts,
                    flats,
                    effs,
                    Pp * P_in_mW,
                    ghostp,
                    flatp,
                    effp,
                    Pw * P_in_mW,
                    ghostw,
                    ghostw_roi,
                    flatw,
                    effw,
                    sol,
                    sol_s,
                    solw,
                    "",
                )
            )
        except ValueError as e:
            rows.append(
                (
                    t,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    None,
                    None,
                    None,
                    str(e),
                )
            )
    return rows


def sigma_sweep(nx, ny, P_in_mW, sigmas):
    return np.array([(s,) + spot_power(nx, ny, s)[:2] for s in sigmas]) * np.array(
        [1.0, P_in_mW, 1.0]
    )


def predistort_sweep(nx, ny, P_in_mW, sigmas, B_avail=None):
    rows = []
    for s in sigmas:
        P, ghost, _flat, _eff, _ = predistort_metrics(nx, ny, s, B_avail=B_avail)
        rows.append((s, P * P_in_mW, ghost))
    return np.array(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nx", type=int, default=8)
    ap.add_argument("--ny", type=int, default=8)
    ap.add_argument("--start", type=int, default=200)
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--pin", type=float, default=1000.0, help="mW into the AOD pair")
    ap.add_argument(
        "--targets", type=float, nargs="*", default=[0.05, 0.1, 0.2, 0.5, 1.0, 1.5]
    )
    ap.add_argument(
        "--sigma-scan",
        action="store_true",
        help="sweep sigma directly instead of hitting power targets",
    )
    ap.add_argument("--sigma-lo", type=float, default=0.2)
    ap.add_argument(
        "--sigma-hi",
        type=float,
        default=None,
        help="default: 98%% of the power-maximizing sigma",
    )
    ap.add_argument("--n-sigma", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--starts", type=int, default=4)
    ap.add_argument(
        "--b-avail",
        type=int,
        default=None,
        help="highest RF channel the AWG/amp chain passes for predistortion "
        "anti-spurs; None = unlimited bandwidth (idealized, near-lossless "
        "predistortion) -- set to your hardware's real cutoff for a "
        "realistic comparison",
    )
    ap.add_argument(
        "--ghost-margin-sites",
        type=int,
        default=3,
        help="power-focused strategy: ghosts beyond this many site-spacings "
        "from the array edge are given zero weight (ignored) so the "
        "optimizer isn't spending suppression budget on spurs outside your "
        "region of interest. Note the worst_ghost_dBc column still reports "
        "the (now unconstrained) global worst -- see the ROI ghost column "
        "for the metric this actually optimizes.",
    )
    ap.add_argument(
        "--flat-max-db",
        type=float,
        default=1.0,
        help="power-focused strategy: max allowed spot-to-spot spread (dB "
        "from the mean) while searching for higher total carrier power at "
        "the same drive budget as the other strategies",
    )
    ap.add_argument("--plot", default="power_scan.svg")
    args = ap.parse_args()

    nx = make_comb(args.nx, args.start, args.step)
    ny = make_comb(args.ny, args.start, args.step)

    sig_sweep = np.linspace(0.1, 1.8, 30)
    sw = sigma_sweep(nx, ny, args.pin, sig_sweep)
    swp = predistort_sweep(nx, ny, args.pin, sig_sweep, B_avail=args.b_avail)
    imax = int(np.argmax(sw[:, 1]))
    print(
        "P_in %.1f mW, %dx%d = %d spots, channels %d..%d step %d"
        % (args.pin, args.nx, args.ny, args.nx * args.ny, nx[0], nx[-1], args.step)
    )
    print("ceiling: %.4f mW/spot at sigma = %.2f\n" % (sw[imax, 1], sw[imax, 0]))

    if args.sigma_scan:
        rows = sigma_range_scan(
            nx,
            ny,
            args.pin,
            args.sigma_lo,
            args.sigma_hi,
            n=args.n_sigma,
            rounds=args.rounds,
            n_starts=args.starts,
            b_avail=args.b_avail,
            pf_margin=args.ghost_margin_sites,
            pf_flat_max=args.flat_max_db,
            pf_starts=args.starts,
        )
        print(
            "%-9s %-11s %-13s %-11s %-11s %-11s %-13s %-11s %-11s %-11s %-13s %-11s %-11s %-11s %-13s %-11s %-11s %-11s %s"
            % (
                "sigma",
                "%max drive",
                "mW/spot",
                "ghost dBc",
                "flat dB",
                "IMD eff %",
                "schr mW",
                "schr dBc",
                "schr flat dB",
                "schr eff %",
                "predist mW",
                "predist dBc",
                "predist flat",
                "predist eff %",
                "pwfocus mW",
                "pwfocus dBc",
                "pwfocus ROI dBc",
                "pwfocus flat",
                "pwfocus eff %",
            )
        )
        for (
            sg,
            ach,
            gh,
            flat,
            eff,
            achs,
            ghs,
            flats,
            effs,
            achp,
            ghp,
            flatp,
            effp,
            achw,
            ghw,
            ghw_roi,
            flatw,
            effw,
            _sol,
            _sol_s,
            _solw,
        ) in rows:
            print(
                "%-9.4f %-11.1f %-13.4f %-11.2f %-11.2f %-11.2f %-13.4f %-11.2f %-11.2f %-11.2f %-13.4f %-11.2f %-11.2f %-11.2f %-13.4f %-11.2f %-11.2f %-11.2f %.2f"
                % (
                    sg,
                    100 * sg / (np.pi / 2),
                    ach,
                    gh,
                    flat,
                    100 * eff,
                    achs,
                    ghs,
                    flats,
                    100 * effs,
                    achp,
                    ghp,
                    flatp,
                    100 * effp,
                    achw,
                    ghw,
                    ghw_roi,
                    flatw,
                    100 * effw,
                )
            )
    else:
        print(
            "%-11s %-9s %-11s %-13s %-11s %-11s %-11s %-13s %-11s %-11s %-11s %-13s %-11s %-11s %-11s %-13s %-11s %-11s %-11s %s"
            % (
                "target mW",
                "sigma",
                "%max drive",
                "achieved mW",
                "ghost dBc",
                "flat dB",
                "IMD eff %",
                "schr mW",
                "schr dBc",
                "schr flat dB",
                "schr eff %",
                "predist mW",
                "predist dBc",
                "predist flat",
                "predist eff %",
                "pwfocus mW",
                "pwfocus dBc",
                "pwfocus ROI dBc",
                "pwfocus flat",
                "pwfocus eff %  note",
            )
        )
        rows = power_scan(
            nx,
            ny,
            args.pin,
            args.targets,
            rounds=args.rounds,
            n_starts=args.starts,
            b_avail=args.b_avail,
            pf_margin=args.ghost_margin_sites,
            pf_flat_max=args.flat_max_db,
            pf_starts=args.starts,
        )
        for (
            t,
            s,
            ach,
            gh,
            flat,
            eff,
            achs,
            ghs,
            flats,
            effs,
            achp,
            ghp,
            flatp,
            effp,
            achw,
            ghw,
            ghw_roi,
            flatw,
            effw,
            _sol,
            _sol_s,
            _solw,
            note,
        ) in rows:
            if note:
                print(
                    "%-11.3f %-9s %-11s %-13s %-11s %-11s %-11s %-13s %-11s %-11s %-11s %-13s %-11s %-11s %-11s %-13s %-11s %-11s %-11s UNREACHABLE (%s)"
                    % (
                        t,
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        note,
                    )
                )
            else:
                print(
                    "%-11.3f %-9.4f %-11.1f %-13.4f %-11.2f %-11.2f %-11.2f %-13.4f %-11.2f %-11.2f %-11.2f %-13.4f %-11.2f %-11.2f %-11.2f %-13.4f %-11.2f %-11.2f %-11.2f %.2f"
                    % (
                        t,
                        s,
                        100 * s / (np.pi / 2),
                        ach,
                        gh,
                        flat,
                        100 * eff,
                        achs,
                        ghs,
                        flats,
                        100 * effs,
                        achp,
                        ghp,
                        flatp,
                        100 * effp,
                        achw,
                        ghw,
                        ghw_roi,
                        flatw,
                        100 * effw,
                    )
                )

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.colors import LogNorm, Normalize
        from matplotlib.ticker import MultipleLocator

        ok = (
            np.array([(r[0], r[1], r[2]) for r in rows])
            if args.sigma_scan
            else np.array([r for r in rows if not r[-1]], dtype=object)
        )
        if args.sigma_scan:
            sg_last, sol, sol_s, solw = (
                rows[-1][0],
                rows[-1][-3],
                rows[-1][-2],
                rows[-1][-1],
            )
        else:
            last = next((r for r in reversed(rows) if not r[-1]), None)
            sg_last, sol, sol_s, solw = (
                (last[1], last[-4], last[-3], last[-2])
                if last is not None
                else (None, None, None, None)
            )

        def fields_from_sol(sol):
            """Px, Py power arrays for a (a, phi) tone solution."""
            axo, phix, ayo, phiy = sol
            Mx, My = suggest_M(nx, axo), suggest_M(ny, ayo)
            return line_spectrum(nx, axo, phix, Mx)[1], line_spectrum(
                ny, ayo, phiy, My
            )[1]

        def camera_image(axis, Px, Py, title):
            """Simulated focal-plane image: log-scale intensity over a
            padded region so ghosts outside the array footprint show up,
            like demo_2d's A[0,0] panel."""
            pad = 2 * args.step * max(args.nx, args.ny)
            bx = np.arange(max(nx.min() - pad, 1), nx.max() + pad)
            by = np.arange(max(ny.min() - pad, 1), ny.max() + pad)
            I = np.outer(Py[by], Px[bx])
            I /= I.max()
            m = axis.imshow(
                np.maximum(I, 1e-8),
                norm=LogNorm(1e-6, 1),
                cmap="inferno",
                origin="lower",
                extent=[bx[0], bx[-1], by[0], by[-1]],
                aspect="auto",
                interpolation="nearest",
            )
            axis.add_patch(
                plt.Rectangle(
                    (nx.min() - args.step / 2, ny.min() - args.step / 2),
                    (args.nx - 1) * args.step + args.step,
                    (args.ny - 1) * args.step + args.step,
                    fill=False,
                    ec="cyan",
                    lw=1.4,
                    ls="--",
                )
            )
            axis.set(xlabel="x channel", ylabel="y channel", title=title)
            plt.colorbar(m, ax=axis, fraction=0.046, label="rel. intensity")

        fig = plt.figure(figsize=(19, 8.8))
        gs = fig.add_gridspec(2, 12)
        ax00 = fig.add_subplot(gs[0, 0:6])
        ax01 = fig.add_subplot(gs[0, 6:12])
        ax10 = fig.add_subplot(gs[1, 0:3])
        ax11 = fig.add_subplot(gs[1, 3:6])
        ax12 = fig.add_subplot(gs[1, 6:9])
        ax13 = fig.add_subplot(gs[1, 9:12])
        (l1,) = ax00.plot(
            sw[:, 0],
            sw[:, 1],
            "-",
            color="tab:green",
            label="%dx%d schroeder comb" % (args.nx, args.ny),
        )
        ax00.axvline(sw[imax, 0], color="gray", ls=":")
        if len(ok):
            ax00.plot(
                ok[:, 0] if args.sigma_scan else [r[1] for r in ok],
                ok[:, 1] if args.sigma_scan else [r[2] for r in ok],
                "r*",
                ms=11,
                label="optimized",
            )
            good_rows = rows if args.sigma_scan else [r for r in rows if not r[-1]]
            ax00.plot(
                [r[0] for r in good_rows]
                if args.sigma_scan
                else [r[1] for r in good_rows],
                [r[13] for r in good_rows]
                if args.sigma_scan
                else [r[14] for r in good_rows],
                "bs",
                ms=8,
                label="power-focused",
            )
        ax00.set(
            xlabel=r"$\sigma$ [rad]",
            ylabel="mW / spot",
            title="power ceiling %.3f mW at $\\sigma$=%.2f"
            % (sw[imax, 1], sw[imax, 0]),
        )
        ax00.grid(alpha=0.3)
        (l2,) = ax00.plot(
            sig_sweep,
            args.pin * np.sin(sig_sweep) ** 2 / (args.nx * args.ny),
            "k--",
            lw=1.2,
            label=r"single tone $\sin^2\sigma$ / N spots",
        )
        # Bussgang/Gaussian approx: for u~N(0,sigma^2) the linear ("carrier")
        # gain of sin(u) is g=exp(-sigma^2/2), so per-axis carrier power is
        # g^2*sigma^2 = sigma^2*exp(-sigma^2) (matches demo_2d's reference
        # curve). The 2D field factorizes, so mean per-spot power is the
        # product of the two (identical) per-axis means, each per-axis mean
        # being that carrier power divided by the axis's tone count.
        (l3,) = ax00.plot(
            sig_sweep,
            args.pin
            * (sig_sweep**2 * np.exp(-(sig_sweep**2))) ** 2
            / (args.nx * args.ny),
            "-.",
            color="tab:orange",
            lw=1.2,
            label=r"Gaussian approx $(\sigma^2e^{-\sigma^2})^2$ / N spots",
        )
        b_label = "unlimited" if args.b_avail is None else str(args.b_avail)
        (l4,) = ax00.plot(
            swp[:, 0],
            swp[:, 1],
            "-",
            color="tab:red",
            lw=1.5,
            label="predistort (B_avail=%s)" % b_label,
        )
        ax00.legend(fontsize=8)
        sig_norm = Normalize(vmin=float(sw[:, 0].min()), vmax=float(sw[:, 0].max()))
        cmap = plt.cm.viridis
        pts = np.array([sw[:, 1], sw[:, 2]]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=cmap, norm=sig_norm)
        lc.set_array(0.5 * (sw[:-1, 0] + sw[1:, 0]))
        lc.set_linewidth(2)
        ax01.add_collection(lc)
        ax01.plot(swp[:, 1], swp[:, 2], "-", color="tab:red", lw=1.5, alpha=0.7)
        ax01.autoscale_view()

        if args.sigma_scan:
            sig_v = np.array([r[0] for r in rows])
            xo, yo = np.array([r[1] for r in rows]), np.array([r[2] for r in rows])
            xs, ys = np.array([r[5] for r in rows]), np.array([r[6] for r in rows])
            xp, yp = np.array([r[9] for r in rows]), np.array([r[10] for r in rows])
            xw, yw = np.array([r[13] for r in rows]), np.array([r[14] for r in rows])
        else:
            good = [r for r in rows if not r[-1]]
            sig_v = np.array([r[1] for r in good])
            xo, yo = np.array([r[2] for r in good]), np.array([r[3] for r in good])
            xs, ys = np.array([r[6] for r in good]), np.array([r[7] for r in good])
            xp, yp = np.array([r[10] for r in good]), np.array([r[11] for r in good])
            xw, yw = np.array([r[14] for r in good]), np.array([r[15] for r in good])
        if len(sig_v):
            sc = ax01.scatter(
                xo,
                yo,
                c=sig_v,
                cmap=cmap,
                norm=sig_norm,
                marker="*",
                s=180,
                edgecolors="k",
                linewidths=0.6,
                label="optimized",
                zorder=5,
            )
            ax01.scatter(
                xs,
                ys,
                c=sig_v,
                cmap=cmap,
                norm=sig_norm,
                marker="D",
                s=55,
                edgecolors="k",
                linewidths=0.6,
                label="schroeder",
                zorder=5,
            )
            ax01.scatter(
                xp,
                yp,
                c=sig_v,
                cmap=cmap,
                norm=sig_norm,
                marker="^",
                s=90,
                edgecolors="k",
                linewidths=0.6,
                label="predistort",
                zorder=5,
            )
            ax01.scatter(
                xw,
                yw,
                c=sig_v,
                cmap=cmap,
                norm=sig_norm,
                marker="s",
                s=70,
                edgecolors="k",
                linewidths=0.6,
                label="power-focused",
                zorder=5,
            )
            fig.colorbar(sc, ax=ax01, fraction=0.046, label=r"$\sigma$ [rad]")
            ax01.legend(fontsize=8, loc="lower right")
        ax01.set(xlabel="mW / spot", ylabel="worst ghost [dBc]", title="the price list")
        ax01.yaxis.set_major_locator(MultipleLocator(10))
        ax01.yaxis.set_minor_locator(MultipleLocator(5))
        ax01.grid(which="major", alpha=0.4)
        ax01.grid(which="minor", alpha=0.15)

        if sol is not None:
            Px_o, Py_o = fields_from_sol(sol)
            Px_s, Py_s = fields_from_sol(sol_s)
            Px_w, Py_w = fields_from_sol(solw)
            _, _, _, _, (Px_p, Py_p) = predistort_metrics(
                nx, ny, sg_last, B_avail=args.b_avail, want_fields=True
            )
            camera_image(ax10, Px_o, Py_o, "optimized")
            camera_image(ax11, Px_s, Py_s, "schroeder")
            camera_image(ax12, Px_p, Py_p, "predistort (B_avail=%s)" % b_label)
            camera_image(ax13, Px_w, Py_w, "power-focused")
        else:
            ax10.set_title("no successful result to plot")
            ax11.set_title("no successful result to plot")
            ax12.set_title("no successful result to plot")
            ax13.set_title("no successful result to plot")

        plt.tight_layout()
        plt.savefig(args.plot)
        print("\nwrote", args.plot)


if __name__ == "__main__":
    main()
