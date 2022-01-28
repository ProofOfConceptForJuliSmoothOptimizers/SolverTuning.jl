macro benchmark_with_result(args...)
  _, params = prunekwargs(args...)
  tmp = gensym()
  return esc(quote
    local $tmp = $BenchmarkTools.@benchmarkable $(args...)
    $BenchmarkTools.warmup($tmp)
    $(hasevals(params) ? :() : :($BenchmarkTools.tune!($tmp)))
    $run_with_return_value($tmp)
  end)
end

run_with_return_value(
  b::Benchmark,
  p::Parameters = b.params;
  progressid = nothing,
  nleaves = NaN,
  ndone = NaN,
  kwargs...,
) = run_result(b, p; kwargs...)
