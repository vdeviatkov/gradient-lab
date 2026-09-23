#!/usr/bin/env bash
# Times a clean build of the C++ library and its executables, and writes the result next to the
# run-time benchmarks.
#
# Compilation time is not something the benchmarked program can measure about itself: by the time
# it runs, it has already been built. It is measured here instead, in the one place that sees a
# build from nothing, and the run-time figures say which build they refer to.
set -euo pipefail

build_directory="${1:-build-timing}"
output_directory="${ML_SCRATCH_BENCHMARK_DIR:-results/benchmarks}"
jobs="${ML_SCRATCH_BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"

rm -rf "${build_directory}"
mkdir -p "${output_directory}"

configure_start=$(date +%s.%N)
cmake -S . -B "${build_directory}" -DCMAKE_BUILD_TYPE=Release > /dev/null
configure_end=$(date +%s.%N)

library_start=$(date +%s.%N)
cmake --build "${build_directory}" --target ml_scratch_cpp --parallel "${jobs}" > /dev/null
library_end=$(date +%s.%N)

everything_start=$(date +%s.%N)
cmake --build "${build_directory}" --parallel "${jobs}" > /dev/null
everything_end=$(date +%s.%N)

elapsed() { echo "$2 - $1" | bc -l; }

cat > "${output_directory}/cpp_build.json" <<JSON
{
  "implementation": "cpp",
  "build_type": "Release",
  "parallel_jobs": ${jobs},
  "date": "$(date +%Y-%m-%d)",
  "configure_seconds": $(elapsed "${configure_start}" "${configure_end}"),
  "library_seconds": $(elapsed "${library_start}" "${library_end}"),
  "remaining_targets_seconds": $(elapsed "${everything_start}" "${everything_end}")
}
JSON

rm -rf "${build_directory}"
echo "wrote ${output_directory}/cpp_build.json"
cat "${output_directory}/cpp_build.json"
