Notes on how to use QuTiP's benchmark scripts
=============================================

Comparing QuTiP to qotoolbox and matlab
----------------------------------------

* Use `matlab-benchmarks.py` to generate `matlab-benchmarks.json`:

        $ python matlab_benchmarks.py -o matlab-benchmarks.json

* Use `qutip-benchmarks.py` to generate `qutip-benchmarks.json`:

        $ python qutip_benchmarks.py -o qutip-benchmarks.json

* Use `benchmark_comparison.py` to generate `benchmark-data.json` which is used by
  the d3 script in benchmark.html. It should take two arguments specifying which 
  benchmark runs to compare::

        $ python benchmark_comparison.py -i qutip-benchmarks.json -r matlab-benchmarks.json -o benchmark_data.json

* The html file and d3 scripts uses benchmark_data.json to render the comparison
  graphics.


Comparing different versions of QuTiP
-------------------------------------

Now we can also compare different versions of QuTiP, like this:

    # install qutip 2.2.0
    $ python qutip_benchmarks.py -o qutip-benchmarks-2.2.0.json
    # install qutip 2.3.0-dev
    $ python qutip_benchmarks.py -o qutip-benchmarks-2.3.0-dev.json
    # generate the comparison data, with 2.2.0 as reference
    $ python benchmark_comparison.py -i qutip-benchmarks-2.3.0-dev.json -r qutip-benchmarks-2.2.0.json -o benchmark_data.json

