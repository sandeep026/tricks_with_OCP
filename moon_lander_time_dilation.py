import casadi as cs
def moon_lander():
    N = 25

    opti = cs.Opti()
    nx = 2
    nu = 1
    W = opti.variable(nx + nu + 1, N)

    xnp1 = opti.variable(nx, 1)
    tnp1 = opti.variable(1)
    X = cs.hcat([W[0:nx, :], xnp1])
    U = W[nx : nx + nu, :]
    t = cs.hcat([W[nx + nu : nx + nu + 1, :], tnp1])

    opti.set_linear_scale(t, 8 * cs.DM.ones(t.shape))
    opti.set_linear_scale(
        X, cs.vcat([10 * cs.DM.ones(1, N + 1), 2 * cs.DM.ones(1, N + 1)])
    )
    opti.set_linear_scale(U, 3 * cs.DM.ones(U.shape))

    dt = cs.diff(t)
    xs = cs.SX.sym("xs", nx, 1)
    us = cs.SX.sym("us", nu, 1)
    ts = cs.SX.sym("ts")
    t0s = cs.SX.sym("t0s")
    tfs = cs.SX.sym("tfs")
    dae = {
        "x": xs,
        "t": ts,
        "p": cs.vertcat(us, t0s, tfs),
        "ode": (tfs - t0s) * cs.vertcat(xs[1], us - 1.5),
        "quad": (tfs - t0s) * us,
    }
    cvodes = cs.integrator(
        "integrator",
        "collocation",
        dae,
        0,
        1,
        {"simplify": True, "number_of_finite_elements": 2},
    )
    ires = cvodes.map(N, "serial")(x0=X[:, 0:-1], p=cs.vertcat(U, t[:, 0:-1], t[:, 1:]))
    defect = X[:, 1:] - ires["xf"]
    opti.minimize(cs.sum2(ires["qf"]))
    opti.subject_to(X[:, 0] == cs.DM([10, -2]))
    opti.subject_to(X[:, -1] == cs.DM([0] * 2))
    opti.subject_to(t[:, 0] == 0)
    dtn = (t[:, -1] - t[:, 0]) / N
    opti.subject_to(opti.bounded(0.1 * dtn, dt, 2 * dtn))
    opti.subject_to(defect == 0)
    opti.subject_to(opti.bounded(0, U, 3))
    Xg = cs.vertcat(cs.linspace(10, 0, N + 1).T, cs.linspace(-2, 0, N + 1).T)
    Ug = cs.linspace(0, 3, N).T
    tg = cs.linspace(0, 1, N + 1).T
    opti.set_initial(t, tg)
    opti.set_initial(X, Xg)
    opti.set_initial(U, Ug)
    opti.solver(
        "ipopt",
        {
            "expand": False,
        },
    )
    sol = opti.solve()
    # ---- post-processing        ------
    from pylab import plot, step, figure, legend, show, spy

    plot(sol.value(t), sol.value(X[1, :]), "o-", label="speed")
    plot(sol.value(t), sol.value(X[0, :]), "o-", label="pos")
    step(sol.value(t)[0:-1], sol.value(U), "k", label="throttle")
    legend(loc="upper left")
    figure()
    spy(sol.value(cs.jacobian(opti.g, opti.x)))
    figure()
    spy(sol.value(cs.hessian(opti.f + cs.dot(opti.lam_g, opti.g), opti.x)[0]))
    show()
    return sol.value(dt)

moon_lander()