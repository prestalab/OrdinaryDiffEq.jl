using OrdinaryDiffEqCore, OrdinaryDiffEqTsit5, Test

@testset "Formula and committed floating clock agree" begin
    for T in (Float32, Float64), direction in (T(1), T(-1)), adaptive in (false, true)
        t0 = T(3600)
        dt = T(32) * eps(t0)
        tf = t0 + direction * T(3) * dt
        tol = T === Float32 ? T(16) * eps(T) : T(1.0e-12)
        sol = solve(
            ODEProblem((u, p, t) -> one(u), zero(T), (t0, tf)), Tsit5();
            adaptive, dt = direction * dt, dtmax = dt, reltol = T(1.0e-6),
            abstol = T(1.0e-12), save_everystep = true, dense = false
        )
        @test sol.retcode == ReturnCode.Success
        @test sol.t[end] == tf
        @test sol.u[end] ≈ tf - t0 rtol = tol atol = zero(T)
        @test all(abs(sol.t[j] - sol.t[j - 1]) <= dt for j in 2:length(sol.t))
    end
end

@testset "Represented increment respects a fractional-ulp cap" begin
    for T in (Float32, Float64), direction in (T(1), T(-1))
        t0 = T(3600)
        dt = T(1.5) * eps(t0)
        integ = init(
            ODEProblem((u, p, t) -> one(u), zero(T), (t0, t0 + direction)), Tsit5();
            adaptive = false, dt = direction * dt, dtmax = dt,
            save_everystep = false, dense = false
        )
        step!(integ)
        elapsed = integ.t - t0
        @test direction * elapsed > 0
        @test abs(elapsed) <= dt
        @test integ.dt == elapsed
        @test integ.u ≈ elapsed rtol = T(16) * eps(T) atol = zero(T)
    end
end

@testset "Near-stop handling never lengthens the explicit cap" begin
    for direction in (1.0, -1.0), adaptive in (false, true)
        t0 = 3600.0
        dt = 64eps(t0)
        tf = t0 + direction * 80eps(t0)
        sol = solve(
            ODEProblem((u, p, t) -> 1.0, 0.0, (t0, tf)), Tsit5();
            adaptive, dt = direction * dt, dtmax = dt, abstol = 1.0e-18,
            reltol = 1.0e-8, save_everystep = true, dense = false
        )
        @test sol.retcode == ReturnCode.Success
        @test sol.u[end] ≈ tf - t0 rtol = 1.0e-12 atol = 0.0
        @test all(abs(sol.t[j] - sol.t[j - 1]) <= dt for j in 2:length(sol.t))
    end
end

@testset "Distinct neighbouring stops keep their states" begin
    for direction in (1.0, -1.0)
        t0 = 3600.0
        a = t0 + direction * 64eps(t0)
        b = t0 + direction * 65eps(t0)
        tf = t0 + direction * 96eps(t0)
        visits = Tuple{Float64, Float64}[]
        cb = DiscreteCallback(
            (u, t, integrator) -> t == a || t == b,
            function (integrator)
                push!(visits, (integrator.t, integrator.u))
                derivative_discontinuity!(integrator, false)
            end;
            save_positions = (false, false)
        )
        sol = solve(
            ODEProblem((u, p, t) -> 1.0, 0.0, (t0, tf)), Tsit5();
            dt = direction * 64eps(t0), dtmax = 64eps(t0), tstops = [a, b],
            callback = cb, abstol = 1.0e-18, reltol = 1.0e-8
        )
        @test sol.retcode == ReturnCode.Success
        @test first.(visits) == [a, b]
        @test all(isapprox(u, t - t0; rtol = 1.0e-12, atol = 0.0) for (t, u) in visits)
    end
end

@testset "A stop near zero uses a reachable formula endpoint" begin
    for direction in (1.0, -1.0)
        t0 = -direction
        tf = -direction * eps(1.0)^2
        intervals = Tuple{Float64, Float64, Float64}[]
        cb = DiscreteCallback(
            (u, t, integrator) -> true,
            function (integrator)
                push!(intervals, (integrator.tprev, integrator.t, integrator.dt))
                derivative_discontinuity!(integrator, false)
            end;
            save_positions = (false, false)
        )
        sol = solve(
            ODEProblem((u, p, t) -> 0.0, 3.0, (t0, tf)), Tsit5();
            adaptive = false, dt = 2direction, dtmax = 2.0, callback = cb
        )
        @test sol.retcode == ReturnCode.Success
        @test sol.t[end] == tf
        @test sol.u[end] == 3.0
        @test all(a + dt == b && b - a == dt for (a, b, dt) in intervals)
        @test all(direction * (t - tf) <= 0 for t in sol.t)
    end
end

@testset "Callbacks observe the accepted interval, not its proposal" begin
    observations = Tuple{Float64, Float64, Float64}[]
    cb = DiscreteCallback(
        (u, t, integrator) -> t == 1.0,
        function (integrator)
            push!(observations, (integrator.dt, integrator.t - integrator.tprev, integrator(0.5)))
            derivative_discontinuity!(integrator, false)
        end;
        save_positions = (false, false)
    )
    sol = solve(
        ODEProblem((u, p, t) -> 1.0, 1.0, (0.0, 2.0)), Tsit5();
        dt = 5.0, tstops = [1.0], callback = cb, abstol = 1.0e-8, reltol = 1.0e-8
    )
    dt, interval, midpoint = only(observations)
    @test sol.retcode == ReturnCode.Success
    @test dt == interval
    @test midpoint ≈ 1.5 rtol = 1.0e-12
    @test sol.u[end] ≈ 3.0 rtol = 1.0e-12
end

@testset "Rejected near-stop proposals may shrink normally" begin
    t0 = 3600.0
    tf = t0 + 80eps(t0)
    lambda = 1.0e10
    sol = solve(
        ODEProblem((u, p, t) -> -lambda * u, 1.0, (t0, tf)), Tsit5();
        dt = 64eps(t0), dtmax = 64eps(t0), abstol = 1.0e-12, reltol = 1.0e-10
    )
    @test sol.retcode == ReturnCode.Success
    @test sol.stats.nreject > 0
    @test sol.u[end] ≈ exp(-lambda * (tf - t0)) rtol = 5.0e-9
    @test all(abs(sol.t[j] - sol.t[j - 1]) <= 64eps(t0) for j in 2:length(sol.t))
end

@testset "Unrepresentable fixed-step progress is a failure" begin
    t0 = 3600.0
    sol = solve(
        ODEProblem((u, p, t) -> 1.0, 0.0, (t0, t0 + 64eps(t0))), Tsit5();
        adaptive = false, dt = eps(t0) / 4, dtmax = eps(t0) / 4,
        maxiters = 16, save_everystep = false, dense = false
    )
    @test sol.retcode != ReturnCode.Success
    @test sol.t[end] == t0
    @test sol.u[end] == 0.0
    @test sol.stats.naccept == 0
end

@testset "Native epsilon guard checks the initial proposal" begin
    t0 = 0.5
    tf = t0 + 64eps(t0)
    problem = ODEProblem((u, p, t) -> 1.0, 0.0, (t0, tf))
    admissible = solve(
        problem, Tsit5(); dt = nextfloat(eps(t0)), dtmax = 64eps(t0),
        abstol = 1.0e-18, reltol = 1.0e-8
    )
    @test admissible.retcode == ReturnCode.Success
    @test admissible.u[end] ≈ tf - t0 rtol = 1.0e-12 atol = 0.0
    rejected = solve(
        problem, Tsit5(); dt = eps(t0), dtmax = 64eps(t0),
        abstol = 1.0e-18, reltol = 1.0e-8
    )
    @test rejected.retcode == ReturnCode.Unstable
    @test rejected.stats.naccept == 0
end
