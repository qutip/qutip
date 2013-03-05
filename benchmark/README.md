Note on how to use qutip's benchmark scripts
--------------------------------------------

* Use matlab-benchmarks.py to generate matlab-benchmarks.json

    $ python matlab_benchmark.py -o matlab-benchmarks.json

* Use qutip-benchmarks.py to generate qutip-benchmarks.json::

    $ python qutip_benchmark.py -o qutip-benchmarks.json

* Use benchmark-ratio.py to generate benchmark-data.js which is used by the d3
  script in benchmark.html. It should take two arguments specifying which 
  benchmark runs to compare::

    python benchmark-ration.py qutip-benchmark.json matlab-benchmark.json

* The html file and d3 scripts uses benchmark_data.js to render the comparison
  graphics.
