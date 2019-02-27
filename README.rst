################################################################################
egp
################################################################################

Python code used and developed for my cosmology research


Conventions
-----------

The values in Fields correspond to physical coordinates. The boxlen size spans the full cells on both ends, i.e. it begins at the far end of the [0,0,0] cell and ends at the far end of the [gridsize,gridsize,gridsize] cell. Each cell value corresponds to the coordinates at the center of the cell. This means that the coordinates of cell [0,0,0] are [dx/2,dx/2,dx/2] where dx is the linear size of one cell.

Python versions
---------------

This repository is set up with Python versions:

* 3.4
* 3.5
* 3.6
* 3.7

Add or remove Python versions based on project requirements. `The guide <https://guide.esciencecenter.nl/best_practices/language_guides/python.html>`_ contains more information about Python versions and writing Python 2 and 3 compatible code.

Installation
------------

To install egp, do:

.. code-block:: console

  git clone https://github.com/egpbos/egp.git
  cd egp
  pip install .


Run tests (including coverage) with:

.. code-block:: console

  python setup.py test


Documentation
*************

.. _README:

Include a link to your project's full documentation here.

Contributing
************

If you want to contribute to the development of egp,
have a look at the `contribution guidelines <CONTRIBUTING.rst>`_.

License
*******

Copyright (c) 2009-2019, E. G. Patrick Bos

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.



Credits
*******

This package was created with `Cookiecutter <https://github.com/audreyr/cookiecutter>`_ and the `NLeSC/python-template <https://github.com/NLeSC/python-template>`_.
