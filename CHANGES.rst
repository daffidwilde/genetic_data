History
=======

v0.3.1 (2020-07-28)
-------------------

- Add missing install requirements to `setup.py`.
- Fitness functions must take an instance of `Individual`.
- Fix flaky tests with long runtimes.

v0.3 (2020-07-27)
-----------------

This is the first big improvement in a while and makes the library easier to use
in my opinion. The changes can be summarised as follows:

- The EA has been moved to the `edo.DataOptimiser` class rather than
  `edo.run_algorithm`.
- Individuals are now a fully-stocked class.
- A new class `Family` for handling the subtypes of a `Distribution` class
  externally. Note the move from `edo.families` to `edo.distributions` to
  reflect this and to avoid confusion in the future.
- The `Individual` class is more robust now, taking its fitness as an attribute
  and being able to store and recover itself entirely -- including the subtype
  instances in `Individual.metadata` (a **big** improvement over the
  dictionaries).
- The pseudo-random number generator framework has been totally decentralised.
  This means that stochastic fitness functions can use `np.random.seed` if
  necessary without throwing reproducibility out of the window. `Individual` and
  `Family` instances are provided their own `np.random.RandomState` instances to
  use in sampling, and there is a "lead" state used by `DataOptimiser`.
- Minor changes to `README` and other documentation files hosted on GitHub.

v0.2.1 (2019-04-25)
-------------------

- Ignore Dask aux files if they come up
- Fix warning from PyYAML

v0.2 (2019-04-15)
-----------------

- Column distributions can now produce independent versions of themselves (#112)
- Fitness computation is parallelised and cached (#117)
- Results written to disk. Output metadata as dictionaries.

v0.1 (2019-02-05)
-----------------

- Resetting of columns is now based on the original parameter limits
- Minor fixes in documentation and larger code base

v0.0.4a0 (2019-01-31)
---------------------

- Remove unwanted print statements
- Update images in docs
- Add context line to setup.py

v0.0.4 (2019-01-30)
-------------------

- Fix bug in search space compact function.
- Get build to pass.

v0.0.3 (2019-01-30)
-------------------

- Finish documentation.

v0.0.2 (2019-01-30)
-------------------

- First proper release. Documentation to come.
