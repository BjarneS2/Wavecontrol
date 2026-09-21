"""intermodulation_simulation.py
Intermodulation optimizer for multi-tone AOD driving.

Bragg-regime AOD, assumed to have a memoryless odd non-linearity in the
diffracted field relative to the RF drive.

    Diffr. Effic. = sin^2(u) = (1 - cos 2u)/2,

        -->     EField_diff / EField_in = sin( u(t) ),

    Input signal (equally spaced grid):
    u(t) = sum_k a_k cos(2 pi f_k t + phi_k),     f_k = n_k * Delta

    Taylor expansion of the field response (sin(*)) yields:
    sin(u(t)) ~ u(t) - u(t)^3/3! + u(t)^5/5! - ...
    Non-linearity creates third, and higher, order products. This creates
    new spots and spurious frequencies that steal power from the original
    imput signals and redistributes them. For an equally spaced grid these
    intermodulations land on the exact same spots and such on grid products
    cause static depth/intensity errors. Some spots may lie outside of the
    grid and thus steal power or create beat notes on the order of MHz,
    and are time-averaged away if the response of the target is slower than
    MHz - which is not the case for NV-center responses, as the Rabi-rates
    are on the order of MHz. As an
    attempt to reduce these higher order products (dominated by the 3rd),
    one can vary the phases to flatten the response and using the AOD in
    a linear regime and not push to the maximum power. This also goes for
    any amplifier and electronic element in between the drive and AOD that
    can cause these intermodulation products through non-linearities.

A mitigation technique is varying the phases as it to reduce power spikes
that push the hardware into the non-linear regions as a form of reduction
in the resulting output. The crest factor (maximum in one period over the
RMS) is a quantity that probes this.

The optimization is a Multi-start L-BFGS-B.
Breaking this down it is the limited memory version of the Broyden-Fletcher-
Goldfarb-Shanno algorithm for unconstrained non-linear optimization problems.
The B brings back the bounds for the variables, here the amplitudes of the
tones. Multi-start, because the landscape of the objective function contains
many local minima and to find a minimum that is valid more or less globally
one needs to initialize at many different starting positions to find an
optimal set of parameters through many tries. (L-BFGS-B is greedy and will
only find the next local-minimum.)



SYMBOLS
-------
    K        number of tones on one AOD axis
    n_int    integer channel index per tone       (f_k = n_k * Delta)
    a        drive amplitude per tone, radians    (pi/2 = saturation)
    phi      RF phase per tone, radians
    M        FFT length = samples per waveform period
    u[j]     drive waveform sampled at t_j = j/(M*Delta)
    y[j]     diffracted field, y = sin(u)
    Y[b]     rfft(y);  line amplitude A_b = 2|Y_b|/M
    P[b]     optical line power = A_b^2/2 = 2|Y_b|^2/M^2
    w[b]     bin weight: how much distortion in bin b costs us
    g[j]     adjoint field, dJ/dy_j
    h[j]     adjoint through the nonlinearity, h = g * cos(u)
    spots    bin indices carrying a real tweezer (= n_int)
    spurs    occupied bins that are not spots

=======================================================================
"""

import math

import numpy as np


def suggest_M(n_int, a, safety=4.0, min_pow2=12):
    """
    For larger K the number of intermodulation products scales bad. This is
    why we need to choose an FFT length (which hits the products exactly) so
    no significant intermodulation product aliases.
    Through: Fourier coefficients of a phase-modulated carrier ~ Bessel functions.
    We can approximate J_n(z) as proportional to (z/n)^n which decays super-expon.
    for n >> z. We can then pick the extends:
    """
    R_eff = 3.0 * float(np.max(a)) + 5.0
    need = safety * R_eff * float(np.max(np.abs(n_int)))
    return int(2 ** max(min_pow2, math.ceil(math.log2(max(need, 2)))))


def synthesize(n_int, a, phi, M):
    """
    Builds one period of u(t) by inverse FFT.

    Placing (M/2) * a_k * exp(i phi_k) at half-spectrum index n_k makes
    irfft return exactly a_k cos(2 pi n_k j / M + phi_k).
    """
    X = np.zeros(M // 2 + 1, dtype=np.complex128)
    np.add.at(X, n_int, (M / 2.0) * a * np.exp(1j * phi))
    return np.fft.irfft(X, M)  # inverse fourier


def line_spectrum(n_int, a, phi, M=None):
    """
    Exact line spectrum of sin(u).  Returns (A, P) indexed by bin b,
    with A = field amplitude and P = optical power (fraction of P_in).
    """
    M = M or suggest_M(n_int, a)
    Y = np.fft.rfft(np.sin(synthesize(n_int, a, phi, M)))
    A = 2.0 * np.abs(Y) / M
    return A, 0.5 * A**2


def alias_check(n_int, a, phi, M=None):
    """
    The FFT engine is exact ONLY if nothing folds past Nyquist.
    Compare M against 2M; the max relative deviation must be ~1e-15.
    """
    M = M or suggest_M(n_int, a)
    A1, _ = line_spectrum(n_int, a, phi, M)
    A2, _ = line_spectrum(n_int, a, phi, 2 * M)
    nb = A1.size
    ref = max(A1.max(), 1e-30)
    return float(np.max(np.abs(A1 - A2[:nb])) / ref)


# =======                               =======
# Now we build the cost function (and gradient)
# =======                               =======


def make_weights(n_int, M, w_inside=1.0, w_outside=0.05, margin=0):
    """
    Defines the bin weights.  Distortion landing inside the array footprint
    create ghost spots that cause excitations in the NV layer. This would
    be a source of noise and makes SNR more expensive.  Distortion landing
    outside just steals power.  Spot bins themselves get weight 0 here;
    carrier flatness is handled by a separate term.
    """
    nb = M // 2 + 1
    w = np.full(nb, w_outside)
    lo, hi = int(n_int.min()) - margin, int(n_int.max()) + margin
    w[max(lo, 0) : hi + 1] = w_inside
    w[n_int] = 0.0  # spots are not spurs
    w[0] = 0.0  # DC carries no optical meaning here
    w[-1] = 0.0  # Nyquist bin excluded (adjoint identity)
    return w


def cost_and_grad(
    n_int, a, phi, w, M, p=1.0, lam_flat=0.0, p_target=None, want_a_grad=False
):
    """
    J = ( sum_b w_b P_b^p )^(1/p)  +  lam_flat * sum_k (P_{n_k} - target)^2

    Returns (J, dJ/dphi[, dJ/da]).  Four FFTs total, regardless of K.

    p = 1   -> total weighted spur power (smooth, well conditioned)
    p large -> approaches worst-spur (sharp, ill conditioned)
    Anneal p upward during optimization.
    """
    # Generate the input signal:
    u = synthesize(n_int, a, phi, M)
    # Apply the effect of the AOD to the signal:
    y = np.sin(u)
    # fourier transform to get the amplitudes in each created tone:
    Y = np.fft.rfft(y)

    s = 2.0 / M**2  # s is the factor to convert |Y|^2 -> optical power
    P = s * np.abs(Y) ** 2

    # spurious terms are gives as J_p. The objective function J at the p-norm
    # The term is computed by the sum of all spurious powers np.sum(w * P**p)
    # with the weights from before with ^1/p as the gradient dJ / dP
    dJdP = np.zeros_like(P)  # the gradient
    Sp = float(np.sum(w * P**p))  # the sum of the powers inside the tones (weighted)
    if Sp > 0:
        J = Sp ** (1.0 / p)  # objective function
        dJdP += Sp ** (1.0 / p - 1.0) * w * P ** (p - 1.0)  # gradient
    else:
        J = 0.0  # no spurious terms - objective is 0.

    # We also want the powers to be flat among the carrier tones from our input
    # so that the equally spaced grid is also the same amplitude everywhere.
    # lambda is a parameter to tone how strongly weighted the flatness should be
    # during the optimization.
    if lam_flat > 0.0:
        Pc = P[n_int]  # carrier power of tone n_int
        tgt = float(np.mean(Pc)) if p_target is None else float(p_target)  # target pow
        J += lam_flat * float(np.sum((Pc - tgt) ** 2))  # add to the objective
        dJdP[n_int] += 2.0 * lam_flat * (Pc - tgt)  # add to the gradient

    # We need to also compute the adjoint since our cost function combines many
    # inputs R^K -> R into one output. Going forward from the
    # phases phi -> synth. inputs u -> outputs y -> fourier trans. Y -> cost J
    # a finite difference approach costs one sweep per input, so K+1 (or 2K+1
    # for central differences) full evaluations to collect all partials. Doing
    # the adjoint directly, which is a backpropagation varying one output
    # instead of one input, yields 2 more FFTs independent of K.

    # First step backwards: J sees the spectrum only through the line powers
    # P_b = s * |Y_b|^2, so dJ/d(conj Y_b) = (dJ/dP_b) * s * Y_b. That is G,
    # the gradient of the cost in the frequency domain. dJdP alone only says
    # how expensive bin b is. Multiplying by Y_b re-attaches the phase of the
    # line that is actually sitting in that bin, since only its amplitude
    # entered the cost. So G has the same shape as Y and can be read as the
    # "spectrum of the complaint" the cost function is making.
    G = (dJdP * s) * Y
    G[0] = 0.0
    G[-1] = 0.0  # keeps the irfft identity exact
    # irfft rebuilds a real signal by mirroring bins 1...M/2-1 onto the negative
    # frequencies (so they count twice), while DC and Nyquist only appear once.
    # Our sum wants the factor 2 everywhere, so drop those two bins - they are
    # weighted to zero anyway. Then M * irfft(G) is exactly the adjoint
    # field g_j = dJ/dy_j: how much the cost changes if the diffracted field at
    # time sample j is nudged.
    g = M * np.fft.irfft(G, M)

    # Second step backwards: y = sin(u) is pointwise, so its jacobian is
    # diagonal with entries cos(u_j) and propagating through it is just an
    # elementwise multiply. h = g * cos(u) is then dJ/du_j, i.e. the
    # sensitivity w.r.t. the drive-waveform instead of the output. One more FFT
    # turns the time-domain sensitivity into a spectrum H, and since u only
    # ever gets contributions from our own tones we only need H at the tone
    # bins: H[n_k] is how much the cost cares about the drive at the frequency
    # of tone k.
    H = np.fft.rfft(g * np.cos(u))
    # H[n_k] is still in the lab phase reference. Multiplying by exp(i phi_k)
    # rotates it into the phase frame of tone k itself, so z_k becomes the
    # derivative w.r.t. the complex drive coefficient c_k = a_k exp(i phi_k).
    # In that complex plane moving radially changes the amplitude and moving
    # tangentially changes the phase, which is why one single z carries both
    # gradients at once.
    z = np.exp(1j * phi) * np.conj(H[n_int])

    # radial part -> dJ/da_k (amplitude gradient), tangential part -> dJ/dphi_k
    # (phase gradient). The extra -a_k is the jacobian of the polar map, since
    # du/dphi_k = -a_k sin(theta_k).
    dphi = -a * np.imag(z)
    if want_a_grad:
        return J, dphi, np.real(z)
    return J, dphi


# =======                                       =======
# Now we build phase initialization helpers + gauging
# =======                                       =======


def gauge_basis(n_int):
    """
    The cost can have two flat directions:
        phi -> phi + alpha        (global phase)
        phi -> phi + beta * n_k   (time delay / linear ramp)
    Return an orthonormal basis for that null space so we can project it
    out. Without the flat regions we trim down the space for the solver
    to explore. Leaving it in makes the Hessian rank-deficient and wastes
    L-BFGS curvature updates. This may only have a small effect but may
    decrease the iterations.
    """
    V = np.stack(
        [np.ones_like(n_int, dtype=float), np.asarray(n_int, dtype=float)], axis=1
    )
    Q, _ = np.linalg.qr(V)
    return Q


def project_gauge(vec, Q):
    """Remove the components of vec along the flat directions."""
    return vec - Q @ (Q.T @ vec)


def schroeder_phases(K, powers=None):
    """
    Schroeder (1970).  d^2 phi / dn^2 = -2 pi p_n : quadratic phase in
    frequency = linear chirp in time = flat envelope.  Crest factor
    tends to sqrt(2) as K -> infinity. Not valid for only few tones or
    rather not performing as well for few tones, is better than aligned
    tones and roughly as good as random phases (rnd worse on average).
    """
    p = (
        np.full(K, 1.0 / K)
        if powers is None
        else np.asarray(powers, float) / np.sum(powers)
    )
    phi = np.zeros(K)
    for n in range(1, K):
        phi[n] = -2.0 * np.pi * np.sum((n - np.arange(n)) * p[:n])
    return np.mod(phi, 2 * np.pi)


def newman_phases(K):
    """Newman (1965).  Gauge-equivalent to Schroeder for a flat comb."""
    return np.mod(np.pi * np.arange(1, K + 1) ** 2 / K, 2 * np.pi)


def crest_factor(n_int, a, phi, M=None):
    """CF = max|u| / rms(u).  Phases cannot change the rms, only the peak."""
    M = M or suggest_M(n_int, a)
    u = synthesize(n_int, a, phi, M)
    return float(np.max(np.abs(u)) / np.sqrt(np.mean(u**2)))


def metrics(n_int, a, phi, M=None, margin=None, notch=None):
    """
    Everything you would quote in a paper.  Powers are fractions of the
    input optical power.

    IMPORTANT STRUCTURAL FACT (Lecture 2.2): for a uniform comb every
    product lands at f_c + m*Delta, i.e. exactly on the spot lattice.
    So *inside* the array footprint there are no ghost bins at all --
    the in-band damage is entirely carrier contamination, measured by
    `flatness_dB`.  Ghost spots live only outside the footprint.
    That is exactly why NPR (punch a notch, measure what fills it) is
    the right lab metric: it manufactures an in-band ghost bin.

    notch : channel index of a deleted tone.  If given, npr_dB reports
            how far the power that lands in that empty site sits below
            the weakest real spot.
    """
    M = M or suggest_M(n_int, a)
    # We only consider the powers because every number here is a power
    # ratio. P is indexed by bin, so P[b] is the optical power at position
    # b*Delta in the focal plane.
    _, P = line_spectrum(n_int, a, phi, M)

    spots = np.asarray(n_int)
    # Channel spacing in integer units.
    step = int(np.min(np.diff(np.sort(spots)))) if spots.size > 1 else 1
    # How far should we consider ghosts as "near" is set by the margin variable.
    # The outermost third order product is 2*n_max - n_min, which sits exactly
    # one array width beyond the edge (W=K-1 steps). The default here is about
    # 1.5 times W. It covers all third order and part of the fifth order.
    # Anything furhter out is far enough away from the sample that it can be
    # either masked or ignored.
    margin = 3 * step * spots.size // 2 if margin is None else margin

    # Now we markt he spots that we want the light to go into.
    is_spot = np.zeros(P.size, dtype=bool)
    is_spot[spots] = True

    # classify all that are near, while excluding the DC value
    near = np.zeros(P.size, dtype=bool)
    near[max(int(spots.min()) - margin, 1) : int(spots.max()) + margin + 1] = True

    # carrier power
    Pc = P[spots]
    # everything else are ghosts
    ghost = np.where(is_spot, 0.0, P)
    ghost[0] = 0.0
    # for dBc take a reference point - the weakest carrier/spot
    ref = float(Pc.min())

    out = {
        "eta": float(P[1:].sum()),
        "carrier_power": float(Pc.sum()),
        "imd_power": float(ghost.sum()),
        "worst_ghost_dBc": 10 * np.log10(max(ghost.max(), 1e-300) / max(ref, 1e-300)),
        "near_ghost_dBc": 10
        * np.log10(max(float(ghost[near].max()), 1e-300) / max(ref, 1e-300)),
        "flatness_dB": 10 * np.log10(float(Pc.max() / max(Pc.min(), 1e-300))),
        "crest_factor": crest_factor(n_int, a, phi, M),
        "spot_powers": Pc,
    }
    if notch is not None:
        # Noise power ratio == NPR.
        out["npr_dB"] = 10 * np.log10(max(float(P[notch]), 1e-300) / max(ref, 1e-300))
    return out


# =======                   =======
# Now we build the phase optimizer.
# =======                   =======
def optimize_phases(
    n_int,
    a,
    M=None,
    w=None,
    p_schedule=(1.0, 4.0, 12.0),
    n_starts=16,
    maxiter=300,
    lam_flat=0.0,
    seed=0,
    select="worst_ghost_dBc",
    notch=None,
    verbose=False,
):
    """
    Multi-start L-BFGS-B with the analytic adjoint gradient.
    """
    from scipy.optimize import minimize

    n_int = np.asarray(n_int)
    a = np.asarray(a, dtype=float)
    K = n_int.size
    M = M or suggest_M(n_int, a)
    w = make_weights(n_int, M) if w is None else w
    Q = gauge_basis(n_int)
    rng = np.random.default_rng(seed)

    starts = [schroeder_phases(K), newman_phases(K)]
    for i in range(max(n_starts - 2, 0)):
        scale = 0.15 + 1.85 * i / max(n_starts - 3, 1)
        starts.append(schroeder_phases(K) + rng.normal(0, scale, K))

    def run(phi0, p):
        def fg(x):
            J, gphi = cost_and_grad(  # type: ignore
                n_int, a, x, w, M, p=p, lam_flat=lam_flat, want_a_grad=False
            )
            return J, project_gauge(gphi, Q)

        r = minimize(
            fg,
            phi0,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": maxiter, "maxcor": 20, "ftol": 1e-16, "gtol": 1e-14},
        )
        return r.x

    best, best_val = None, np.inf
    for s, phi0 in enumerate(starts):
        phi = np.array(phi0, dtype=float)
        for p in p_schedule:
            phi = run(phi, p)
        mm = metrics(n_int, a, phi, M, notch=notch)
        val = mm[select] if not callable(select) else select(mm)
        if val < best_val:
            best_val, best = val, phi.copy()
        if verbose:
            print(
                "  start %2d -> %7.2f dBc%s"  # noqa: UP031
                % (s, val, "  *" if best_val == val else "")
            )

    return np.mod(best, 2 * np.pi), best_val  # type: ignore


# =====================================================================
#  Now we can also handle the flattening of the amplitudes
# =====================================================================


def flatten_amplitudes(n_int, a0, phi, M=None, target=None, bounds=(1e-4, np.pi)):
    """
    Solve for drive amplitudes that make all TRAP POWERS equal.

    Equal drive does not give equal spot depth: the on-carrier products
    (Lecture 1.4, the 3/2 a_3 A B^2 cross-modulation term) differ per
    site.  This is the in-band damage, and amplitude is the knob for it.
    """
    from scipy.optimize import least_squares

    M = M or suggest_M(n_int, a0)
    tgt = (
        float(np.mean(line_spectrum(n_int, a0, phi, M)[1][n_int]))
        if target is None
        else target
    )

    def resid(x):
        P = line_spectrum(n_int, np.clip(x, *bounds), phi, M)[1][n_int]
        return (P - tgt) / tgt

    sol = least_squares(
        resid,
        np.asarray(a0, float),
        bounds=bounds,
        xtol=1e-14,
        ftol=1e-14,
        max_nfev=200 * len(a0),
    )
    return np.clip(sol.x, *bounds)


# =======                   =======
# Now we build the joint optimizer.
# =======                   =======


def optimize_joint(
    n_int, a0, M=None, rounds=4, lam_flat=1.0, n_starts=16, seed=0, verbose=False
):
    """
    Alternate phase optimization (with a flatness penalty) and amplitude
    pre-compensation. Sequential optimization without the penalty
    oscillates: re-optimizing phases changes the products on the carriers.
    """
    M = M or suggest_M(n_int, a0)
    a = np.asarray(a0, float).copy()
    best, best_score = None, np.inf
    for r in range(rounds):
        phi, _ = optimize_phases(
            n_int,
            a,
            M=M,
            lam_flat=lam_flat,  # non zero otherwise no gain in this
            n_starts=n_starts if r == 0 else 4,
            seed=seed + r,
        )
        a = flatten_amplitudes(n_int, a, phi, M)
        m = metrics(n_int, a, phi, M)
        # alternation is NOT monotone - flattening perturbs the phases'
        # optimum - so keep the best round, do not just return the last.
        score = m["worst_ghost_dBc"] + 10.0 * m["flatness_dB"]
        if score < best_score:
            best_score, best = score, (a.copy(), phi.copy())
        if verbose:
            print(
                "  round %d: ghost %7.2f dBc   flat %6.3f dB   eta %.4f%s"  # noqa: UP031
                % (
                    r,
                    m["worst_ghost_dBc"],
                    m["flatness_dB"],
                    m["eta"],
                    "  *" if best_score == score else "",
                )
            )
    return best


# =======                                               =======
# Now we build the 2D AOD system - 2 independent sets of tones.
# =======                                               =======


def grid_2d(nx, ax, phix, ny, ay, phiy, M=None):
    """
    Two crossed AODs, where the optical field factorizes so the focal
    plane intensity is Px_i * Py_j. The spot non-uniformity is
    separable and flatten each axis delivers a flattened 2D grid. A
    single ghost in x/y will result in a whole ROW or COLUMN of ghosts.
    1D spur suppression is very important especially those multiply too.
    """
    Mx = M or suggest_M(nx, ax)
    My = M or suggest_M(ny, ay)
    Px = line_spectrum(nx, ax, phix, Mx)[1]
    Py = line_spectrum(ny, ay, phiy, My)[1]

    spots = np.outer(Px[nx], Py[ny])

    gx = Px.copy()
    gx[nx] = 0.0
    gx[0] = 0.0
    gy = Py.copy()
    gy[ny] = 0.0
    gy[0] = 0.0

    ref = float(spots.min())
    ghost_cols = np.outer(gx, Py[ny])  # X-spur  times  real Y spots
    ghost_rows = np.outer(Px[nx], gy)  # real X spots  times  Y-spur

    return {
        "spots": spots,
        "flatness_dB": 10 * np.log10(float(spots.max() / max(spots.min(), 1e-300))),
        "worst_ghost_dBc": 10
        * np.log10(
            max(float(ghost_cols.max()), float(ghost_rows.max()), 1e-300)
            / max(ref, 1e-300)
        ),
        "n_ghost_columns": int(np.sum(gx > 1e-6 * ref)),
        "n_ghost_rows": int(np.sum(gy > 1e-6 * ref)),
    }


# def make_comb(K, center_channel, step_channel):
#     """Integer channel indices for a uniform K-tone comb."""
#     k = np.arange(K) - (K - 1) / 2.0
#     return np.round(center_channel + step_channel * k).astype(int)


def make_comb(K, start_channel, step_channel):
    """Integer channel indices for a uniform K-tone comb, lowest tone first."""
    n = start_channel + step_channel * np.arange(K, dtype=int)
    if n.min() < 1:
        raise ValueError(
            "channel indices must be >= 1 (bin 0 is DC); raise start_channel"
        )
    return n


# =======                                                           =======
# Now we work in predistortion and investigate distortion vs efficiency.
# =======                                                           =======


def predistort(n_int, A_tgt, phi, M=None, B_avail=None, clip=1e-3):
    """
    Drive the AOD with u = arcsin(s) so that sin(u) = s exactly.

    s(t) is the TARGET diffracted field -- the clean comb you want.
    Requires max|s| <= 1, so crest factor returns as a hard feasibility
    constraint rather than a soft proxy.

    arcsin(x) = x + x^3/6 + 3x^5/40 + ...   vs   sin(u) = u - u^3/6 + ...
    The cubic terms are equal and opposite: the predistorter emits
    ANTI-SPURS that cancel the AOD's own products.

    B_avail : highest channel index the RF chain passes.  Anti-spurs
              above it are filtered out and their cancellation is lost.
    """
    M = M or suggest_M(n_int, np.asarray(A_tgt) * 2 + 1.0)
    s = synthesize(n_int, np.asarray(A_tgt, float), phi, M)
    pk = np.max(np.abs(s))
    if pk > 1.0 - clip:
        s = s * (1.0 - clip) / pk
    u = np.arcsin(s)
    if B_avail is not None:
        U = np.fft.rfft(u)
        U[int(B_avail) + 1 :] = 0.0
        u = np.fft.irfft(U, M)
    Y = np.fft.rfft(np.sin(u))
    A = 2.0 * np.abs(Y) / M
    return u, A, 0.5 * A**2


def drive_sweep(n_int, phi_fn, a_values, M=None):
    """The efficiency-vs-distortion trade curve. phi_fn(K) -> phases."""
    K = len(n_int)
    out = []
    for av in a_values:
        a = np.full(K, av)
        MM = M or suggest_M(n_int, a)
        m = metrics(n_int, a, phi_fn(K), MM)
        out.append(
            (av, m["eta"], m["worst_ghost_dBc"], m["flatness_dB"], m["crest_factor"])
        )
    return np.array(out)


# =====================================================================
#  BLOCK 8 -- verification suite
# =====================================================================


def check_drive(n_int, a, phi=None, warn_peak=1.2):
    """Peak drive in radians. Beyond ~1.2 rad the AOD is badly nonlinear;
    beyond pi/2 you are past the first maximum of sin^2 and the carriers
    start being destroyed rather than merely distorted."""
    M = suggest_M(n_int, a)
    phi = schroeder_phases(len(a)) if phi is None else phi
    u = synthesize(n_int, a, phi, M)
    pk, rms = float(np.max(np.abs(u))), float(np.sqrt(np.mean(u**2)))
    if pk > warn_peak:
        print(
            "WARNING: peak drive %.2f rad (rms %.2f, CF %.2f). "
            "Scale a by ~%.3f, or use a_k = c/sqrt(K)."
            % (pk, rms, pk / rms, warn_peak / pk)
        )
    return pk, rms


def verify(seed=0):
    """V1 aliasing, V2 Parseval, V3 gauge invariance, V4 adjoint gradient."""
    rng = np.random.default_rng(seed)
    n = make_comb(5, 60, 4)
    a = rng.uniform(0.3, 1.0, 5)
    phi = rng.uniform(0, 2 * np.pi, 5)
    M = suggest_M(n, a)
    ok = True

    r = alias_check(n, a, phi, M)
    print("V1 aliasing (M vs 2M)      : %.2e" % r)
    ok &= r < 1e-12

    A, P = line_spectrum(n, a, phi, M)
    u = synthesize(n, a, phi, M)
    r = abs(P[1:].sum() - np.mean(np.sin(u) ** 2)) / np.mean(np.sin(u) ** 2)
    print("V2 Parseval                : %.2e" % r)
    ok &= r < 1e-10

    # --- V3: the two gauge symmetries are NOT equally exact.
    #
    # RAMP  phi -> phi + beta*n_k  is a pure time delay, so |Y_b| is
    #       invariant to machine precision, always.
    #
    # GLOBAL phi -> phi + alpha multiplies a term of signed order
    #       N = sum(m_k) by exp(i*N*alpha).  It is exact only if every
    #       bin receives a single value of N.  For a uniform comb
    #       b = N*n_c + dn*sum(k*m_k), so orders N and N+2 collide
    #       whenever 2*n_c/dn is an integer -- then the symmetry is only
    #       approximate, broken at the order-3/order-1 amplitude ratio.
    worst = 0.0
    for _ in range(10):
        be = rng.uniform(-np.pi, np.pi)
        A2, _ = line_spectrum(n, a, phi + be * n, M)
        worst = max(worst, np.max(np.abs(A2 - A)) / A.max())
    print("V3a gauge: ramp (exact)    : %.2e" % worst)
    ok &= worst < 1e-12

    worst = 0.0
    for _ in range(10):
        al = rng.uniform(-np.pi, np.pi)
        A2, _ = line_spectrum(n, a, phi + al, M)
        worst = max(worst, np.max(np.abs(A2 - A)) / A.max())
    nc, dn = float(np.mean(n)), float(np.min(np.diff(np.sort(n))))
    exact = abs(2 * nc / dn - round(2 * nc / dn)) > 1e-9
    tol = 1e-12 if exact else 1e-3
    print(
        "V3b gauge: global (%s) : %.2e  [2n_c/dn = %.2f]"
        % ("exact " if exact else "approx", worst, 2 * nc / dn)
    )
    ok &= worst < tol

    w = make_weights(n, M)
    worst = 0.0
    for p, lf in [(1.0, 0.0), (8.0, 0.0), (1.0, 100.0)]:
        J, gp, ga = cost_and_grad(n, a, phi, w, M, p=p, lam_flat=lf, want_a_grad=True)
        for vec, g, which in [(phi, gp, "p"), (a, ga, "a")]:
            num = np.zeros(len(vec))
            h = 1e-6
            for k in range(len(vec)):
                d = np.zeros(len(vec))
                d[k] = h
                args = lambda v: (n, a, v, w, M) if which == "p" else (n, v, phi, w, M)
                num[k] = (
                    cost_and_grad(*args(vec + d), p=p, lam_flat=lf)[0]
                    - cost_and_grad(*args(vec - d), p=p, lam_flat=lf)[0]
                ) / (2 * h)
            worst = max(
                worst, np.max(np.abs(num - g)) / max(np.max(np.abs(num)), 1e-30)
            )
    print("V4 adjoint gradient        : %.2e" % worst)
    ok &= worst < 1e-5

    print("\n%s" % ("ALL CHECKS PASSED" if ok else "*** FAILURE ***"))
    return ok


# =====================================================================
#  BLOCK 9 -- demos
# =====================================================================


def demo(K=64, a0=None, outfile="imd_demo.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if a0 is None:
        a0 = 1 / K

    n = make_comb(K, 70, 4)
    M = suggest_M(n, np.full(K, a0))
    cases = {}
    cases["aligned"] = (np.full(K, a0), np.zeros(K))
    cases["Schroeder"] = (np.full(K, a0), schroeder_phases(K))
    a_opt, phi_opt = optimize_joint(n, np.full(K, a0), M=M, rounds=3, n_starts=50)  # type: ignore
    cases["optimized"] = (a_opt, phi_opt)

    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))

    for name, (a, phi) in cases.items():
        A, P = line_spectrum(n, a, phi, M)
        b = np.arange(P.size)
        s = (b > 0) & (b < 3.2 * n.max()) & (P > 1e-12)
        ax[0, 0].plot(b[s], 10 * np.log10(P[s]), ".", ms=3, label=name, alpha=0.8)
    ax[0, 0].plot(
        n,
        10 * np.log10(line_spectrum(n, *cases["optimized"], M)[1][n]),
        "o",
        ms=7,
        mfc="none",
        label="spots",
    )
    ax[0, 0].set(
        xlabel="bin index (∝ position)",
        ylabel="line power [dB of $P_{in}$]",
        title="line spectrum",
        ylim=(-90, 5),
    )
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].grid(alpha=0.3)
    ax[0, 0].set_xlim(0, 300)

    for name, (a, phi) in cases.items():
        u = synthesize(n, a, phi, M)
        ax[0, 1].plot(np.linspace(0, 1, 400), u[: M : M // 400][:400], lw=1, label=name)
    for lv in (-np.pi / 2, np.pi / 2):
        ax[0, 1].axhline(lv, color="k", ls=":", lw=0.8)
    ax[0, 1].set(
        xlabel="t / T",
        ylabel="drive u(t) [rad]",
        title="waveform (dotted = saturation)",
    )
    ax[0, 1].legend(fontsize=8)
    ax[0, 1].grid(alpha=0.3)

    av = np.linspace(0.1, 1.0, 16)
    for name, fn in [
        ("aligned", lambda k: np.zeros(k)),
        ("Schroeder", schroeder_phases),
    ]:
        sw = drive_sweep(n, fn, av, M=M)
        ax[1, 0].plot(sw[:, 1], sw[:, 2], "o-", ms=3, label=name)
    m = metrics(n, a_opt, phi_opt, M)
    ax[1, 0].plot(m["eta"], m["worst_ghost_dBc"], "r*", ms=14, label="optimized")
    ax[1, 0].set(
        xlabel="total diffraction efficiency",
        ylabel="worst ghost [dBc]",
        title="the efficiency / distortion trade",
    )
    ax[1, 0].legend(fontsize=8)
    ax[1, 0].grid(alpha=0.3)

    g = grid_2d(n, a_opt, phi_opt, n, a_opt, phi_opt, M=M)
    im = ax[1, 1].imshow(g["spots"] / g["spots"].max(), cmap="magma", vmin=0, vmax=1)
    ax[1, 1].set(
        title="%dx%d spot depths  (flatness %.3f dB)" % (K, K, g["flatness_dB"])
    )
    plt.colorbar(im, ax=ax[1, 1], fraction=0.046)

    plt.tight_layout()
    plt.savefig(outfile, dpi=110)
    for name, (a, phi) in cases.items():
        mm = metrics(n, a, phi, M)
        print(
            "%-10s  eta %.4f   ghost %8.2f dBc   flat %6.3f dB   CF %.3f"
            % (
                name,
                mm["eta"],
                mm["worst_ghost_dBc"],
                mm["flatness_dB"],
                mm["crest_factor"],
            )
        )
    print(
        "\n2D %dx%d: flatness %.4f dB, worst ghost %.2f dBc, %d ghost columns + %d rows"
        % (
            K,
            K,
            g["flatness_dB"],
            g["worst_ghost_dBc"],
            g["n_ghost_columns"],
            g["n_ghost_rows"],
        )
    )
    return outfile


def sigma_to_a(K, sigma):
    """Equal-power tones with total rms drive sigma (radians)."""
    return np.full(K, sigma * np.sqrt(2.0 / K))


def demo_2d(
    Nx=8,
    Ny=8,
    sigma=0.85,
    start=90,
    step=5,
    rounds=2,
    n_starts=4,
    outfile="imd_2d.png",
):
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    nx, ny = make_comb(Nx, start, step), make_comb(Ny, start, step)
    ax0, ay0 = sigma_to_a(Nx, sigma), sigma_to_a(Ny, sigma)
    Mx, My = suggest_M(nx, ax0), suggest_M(ny, ay0)

    axo, phix = optimize_joint(nx, ax0, M=Mx, rounds=rounds, n_starts=n_starts)  # type: ignore
    ayo, phiy = optimize_joint(ny, ay0, M=My, rounds=rounds, n_starts=n_starts)  # type: ignore

    Px = line_spectrum(nx, axo, phix, Mx)[1]
    Py = line_spectrum(ny, ayo, phiy, My)[1]
    g = grid_2d(nx, axo, phix, ny, ayo, phiy)

    pad = 2 * step * max(Nx, Ny)
    bx = np.arange(max(nx.min() - pad, 1), nx.max() + pad)
    by = np.arange(max(ny.min() - pad, 1), ny.max() + pad)
    I = np.outer(Py[by], Px[bx])
    I /= I.max()

    fig, A = plt.subplots(2, 2, figsize=(12.5, 9.5))

    m = A[0, 0].imshow(
        np.maximum(I, 1e-8),
        norm=LogNorm(1e-6, 1),
        cmap="inferno",
        origin="lower",
        extent=[bx[0], bx[-1], by[0], by[-1]],
        aspect="auto",
        interpolation="nearest",
    )
    A[0, 0].add_patch(
        plt.Rectangle(  # type: ignore
            (nx.min() - step / 2, ny.min() - step / 2),
            (Nx - 1) * step + step,
            (Ny - 1) * step + step,
            fill=False,
            ec="cyan",
            lw=1.4,
            ls="--",
        )
    )
    A[0, 0].set(
        xlabel="x channel",
        ylabel="y channel",
        title="focal plane, log scale (cyan = array footprint)",
    )
    plt.colorbar(m, ax=A[0, 0], fraction=0.046, label="rel. intensity")

    S = np.outer(Py[ny], Px[nx])
    m = A[0, 1].imshow(
        S / S.max(),
        cmap="viridis",
        origin="lower",
        vmin=0.99 * (S / S.max()).min(),
        vmax=1,
    )
    A[0, 1].set(
        xlabel="x index",
        ylabel="y index",
        title="%dx%d spot depths  (flatness %.4f dB)" % (Nx, Ny, g["flatness_dB"]),
    )
    plt.colorbar(m, ax=A[0, 1], fraction=0.046)

    for lbl, P, nn, c in [("x", Px, nx, "tab:blue"), ("y", Py, ny, "tab:red")]:
        b = np.arange(P.size)
        s = (b > 0) & (b < 2.6 * nn.max()) & (P > 1e-14)
        A[1, 0].plot(
            b[s],
            10 * np.log10(P[s] / P[nn].min()),
            ".",
            ms=3.5,
            color=c,
            label="%s axis" % lbl,
            alpha=0.75,
        )
    A[1, 0].axhline(0, color="k", lw=0.8, ls=":")
    A[1, 0].axvspan(nx.min(), nx.max(), color="cyan", alpha=0.12)
    A[1, 0].set(
        xlabel="channel",
        ylabel="dBc (ref = weakest spot)",
        title="1D line spectra",
        ylim=(-70, 6),
    )
    A[1, 0].legend(fontsize=8)
    A[1, 0].grid(alpha=0.3)

    sig = np.linspace(0.15, 2.0, 24)
    pc, wg = [], []
    for sg in sig:
        a = sigma_to_a(Nx, sg)
        mm = metrics(nx, a, schroeder_phases(Nx), Mx)
        pc.append(mm["carrier_power"])
        wg.append(mm["worst_ghost_dBc"])
    A[1, 1].plot(
        sig, pc, "o-", ms=3, color="tab:green", label=r"carrier power $\eta_c$"
    )
    A[1, 1].plot(
        sig,
        sig**2 * np.exp(-(sig**2)),
        "k--",
        lw=1,
        label=r"$\sigma^2e^{-\sigma^2}$ (Bussgang)",
    )
    A[1, 1].axvline(sigma, color="tab:orange", lw=1.2, label=r"chosen $\sigma$")
    A[1, 1].axvline(1.0, color="gray", ls=":", lw=1, label=r"$\sigma=1$ (max power)")
    A[1, 1].set(xlabel=r"total rms drive $\sigma$ [rad]", ylabel="carrier power")
    A2 = A[1, 1].twinx()
    A2.plot(sig, wg, "s-", ms=3, color="tab:purple")
    A2.set_ylabel("worst ghost [dBc]", color="tab:purple")
    A[1, 1].set_title("drive trade-off (x axis, Schroeder)")
    A[1, 1].legend(fontsize=7, loc="upper left")
    A[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(outfile, dpi=110)

    print("sigma=%.2f  a_x=%.4f  a_y=%.4f" % (sigma, axo.mean(), ayo.mean()))
    for lbl, nn, aa, pp, MM in [("x", nx, axo, phix, Mx), ("y", ny, ayo, phiy, My)]:
        mm = metrics(nn, aa, pp, MM)
        print(
            "  {}: eta {:.4f}  carriers {:.4f}  ghost {:7.2f} dBc  flat {:.4f} dB  CF {:.3f}".format(
                lbl,
                mm["eta"],
                mm["carrier_power"],
                mm["worst_ghost_dBc"],
                mm["flatness_dB"],
                mm["crest_factor"],
            )
        )
    print(
        "2D %dx%d: flatness %.4f dB  worst ghost %.2f dBc  "  # noqa: UP031
        "power/spot %.3e  ghost cols %d rows %d"
        % (
            Nx,
            Ny,
            g["flatness_dB"],
            g["worst_ghost_dBc"],
            S.mean(),
            g["n_ghost_columns"],
            g["n_ghost_rows"],
        )
    )
    return outfile


if __name__ == "__main__":
    import sys

    if "--verify" in sys.argv:
        verify()
    else:
        # print("wrote", demo(K=8, a0=0.6))  # 0.15))
        print("wrote", demo_2d())
