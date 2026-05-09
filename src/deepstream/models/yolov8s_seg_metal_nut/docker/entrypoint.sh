#!/usr/bin/env bash
# Container entrypoint. Stages:
#   build-engine   : trtexec INT8 build + BS=1/BS=4 benchmarks
#   prep-loops     : encode metal_nut/test PNGs into 4 loop video sources
#   mock-factory   : single-stream pyservicemaker demo
#   bench-multi    : 4-stream deepstream-app benchmark with **PERF: parsing
#   report         : 5 charts + HTML + PDF
#   all            : run all above in order
set -euo pipefail

cd /app

CMD="${1:-all}"
shift || true

run_build_engine () { bash /app/scripts/build_engine.sh "$@"; }
run_prep_loops   () { bash /app/scripts/prep_test_loops.sh "$@"; }
run_mock_factory () { python3 /app/scripts/run_pyservicemaker_mock_factory.py \
                        --images-root "${IMAGES_ROOT:-/data/mvtec/metal_nut/test}" \
                        --out         "${MOCK_OUT:-/app/samples/metal_nut_mock.ogv}" \
                        --fps         "${MOCK_FPS:-10}"; }
run_bench_multi  () { bash /app/scripts/bench_multistream.sh; }
run_report       () { python3 /app/scripts/generate_report.py "$@"; }

case "$CMD" in
  build-engine)   run_build_engine "$@" ;;
  prep-loops)     run_prep_loops "$@" ;;
  mock-factory)   run_mock_factory "$@" ;;
  bench-multi)    run_bench_multi "$@" ;;
  report)         run_report "$@" ;;
  all)
    run_build_engine
    run_prep_loops
    run_bench_multi
    run_report
    ;;
  bash|sh) exec /bin/bash ;;
  *)
    echo "Usage: entrypoint.sh {build-engine|prep-loops|mock-factory|bench-multi|report|all|bash}"
    exit 1
    ;;
esac
