.. QuTiP
   Copyright (C) 2011-2017, Alexander J. G. Pitchford, Paul D. Nation & Robert J. Johansson

.. This file was created using retext 6.1 https://github.com/retext-project/retext

.. _release_distribution:

************************
Release and Distribution
************************

Preamble
++++++++

This document covers the process for managing updates to the current minor release and making new releases.
Within this document, the git remote ``upstream`` refers to the main QuTiP organsiation repository, and ``origin`` refers to your personal fork.

Instructions on how to backport bugfixes to a release branch are detailed in bugfix_.
You need to do this to make changes to a current release that cannot wait until the next minor release, but need to go out in a micro release as soon as possible.

Follow either release_ if you are making a major or minor release, or microrelease_ instead if it is only a bugfix patch to a current release.
For both, then do the following steps in order:

1. docbuild_, to build the documentation
2. deploy_, to build the binary and source versions of the package, and deploy it to PyPI (``pip``)
3. github_, to release the files on the QuTiP GitHub page
4. web_, to update `qutip.org <http://qutip.org/>`_ with the new links and documentation
5. cforge_, to update the conda feedstock, deploying the package to ``conda``


.. _gitwf:

Git workflow
++++++++++++

.. _bugfix:

Apply bug fix to latest release
-------------------------------
Assuming that the bug(s) has been fixed in some commit on the master,
then this bug(s) fix should now be applied to the latest release.
First checkout ``master``, use ``$ git log`` to list the commits,
and copy the hash(es) for the bug fix commit(s) to some temporary file or similar.

Now check out latest version branch, e.g.

If you have checked out this branch previously, then ::

    $ git checkout qutip-4.0.X
    $ git pull upstream qutip-4.0.X

Otherwise ::

    $ git fetch upstream
    $ git checkout -b qutip-4.0.X upstream/qutip-4.0.X
    $ git push -u origin qutip-4.0.X

Create a branch for the patch ::

    $ git checkout -b patch4.0-fix_bug123 qutip-4.0.X

Pick the commit(s) to be applied to the release.
Using the commit hash(es) copied earlier, cherry pick them into the current bug fix branch, e.g. ::

    $ git cherry-pick 69d1641239b897eeca158a93b121553284a29ee1

for further info see https://www.kernel.org/pub/software/scm/git/docs/git-cherry-pick.html

push changes to your fork ::

    $ git push --set-upstream origin patch4.0-fix_bug123

Make a Pull Request to the latest release branch on Github. 
That is make a PR from the bug fix branch to the release branch (not the master), e.g. `qutip-4.0.X`

Merge this PR when the tests have passed.

.. _microrelease:

Create a new micro release
--------------------------

Commit a change to the ``VERSION`` file, setting it to the new version.
The only change should be in the third identifier, i.e. if the previous version was 4.5.2, then the next micro release must be 4.5.3.
It is ok to have two-digit identifiers; 4.6.10 is the next number after 4.6.9.
The file should contain only the version number in this format, with no extra characters (the automatic line-break at the end is fine).

.. _release:

Create a new minor or major release
-----------------------------------

Create a new branch on the ``qutip/qutip`` repository using GitHub, e.g. 'qutip-4.1.X', beginning at the commit you want to use as the base of the release.
This will likely be something fairly recent on the ``master`` branch.
See `the GitHub help pages <https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-and-deleting-branches-within-your-repository#creating-a-branch>`_ for more information.
Checkout the branch and push to your fork ::

    $ git fetch upstream
    $ git checkout -b qutip-4.1.X upstream/qutip-4.1.X
    $ git push -u origin qutip-4.1.X

Create a new branch from this, e.g. ::

    $ git checkout -b 4.1-release_ready qutip-4.1.X

First change the ``VERSION`` file to contain the new version number.
A major release increments the first number, while a minor release increments the second.
All numbers after the change digit are reset to 0, so the next minor release after 4.5.3 is 4.6.0, and the next major release after either of these is 5.0.0.
The file should contain only the version number in this format, with no extra characters (the automatic line-break at the end is fine).

Next edit ``setup.cfg``.
Change the "Development Status" line in the ``classifiers`` section to ::

    Development Status :: 5 - Production/Stable

Commit both changes, and then push them to your fork ::

    $ git push --set-upstream origin 4.1-release_ready

Make a Pull Request to the release branch.

The "Development Status" of ``master`` should remain ::

    Development Status :: 2 - Pre-Alpha

because it is never directly released.
The ``VERSION`` file on ``master`` should reflect the last major or minor release.


.. _docbuild:

Documentation build
+++++++++++++++++++

Documentation should be rebuilt for a minor or major release.
If there have been any documentation updates as part of a micro release, then it should also be built for this.
The documentation repository is ``qutip/qutip-doc``.

Ensure that the following steps are complete:

- The version should be changed in ``conf.py``.
- Update ``api_doc/classes.rst`` for any new / deleted classes.
- Update ``api_doc/functions.rst`` for any new / deleted functions.
- Update ``changelog.rst`` including all changes that are going into the new release.

Then, fully rebuild the QuTiP documentation using `the guide in the documentation README <https://github.com/qutip/qutip-doc/blob/master/README.md>`_.

.. _deploy:

Build release distribution and deploy
+++++++++++++++++++++++++++++++++++++

This step builds the source (sdist) and binary (wheel) distributions, and uploads them to PyPI (pip).
You will also be able to download the built files yourself in order to upload them to the QuTiP website.

Build and deploy
----------------

This is handled entirely by a GitHub Action.
Go to the `"Actions" tab at the top of the QuTiP code repository <https://github.com/qutip/qutip/actions>`_.
Click on the "Build wheels, optionally deploy to PyPI" action in the left-hand sidebar.
Click the "Run workflow" dropdown in the header notification; it should look like the image below.

.. image:: /figures/release_guide_run_build_workflow.png

- Use the drop-down menu to choose the branch or tag you want to release from.
  This should be called ``qutip-4.5.X`` or similar, depending on what you made earlier.
  This must *never* be ``master``.
- To make the release to PyPI, type the branch name (e.g. ``qutip-4.5.X``) into the "Confirm chosen branch name [...]" field.
  You *may* leave this field blank to skip the deployment and only build the package.
- (Special circumstances) If for some reason you need to override the version number (for example if the previous deployment to PyPI only partially succeeded), you can type a valid Python version identifier into the "Override version number" field.
  You probably do not need to do this.
  The mechanism is designed to make alpha-testing major upgrades with nightly releases easier.
  For even a bugfix release, you should commit the change to the ``VERSION`` file.
- Click the lower "Run workflow" to perform the build and deployment.

At this point, the deployment will take care of itself.
It should take between 30 minutes and an hour, after which the new version will be available for install by ``pip install qutip``.
You should see the new version appear on `QuTiP's PyPI page <https://pypi.org/project/qutip>`_.

Download built files
--------------------

When the build is complete, click into its summary screen.
This is the main screen used to both monitor the build and see its output, and should look like the below image on a success.

.. image:: /figures/release_guide_after_workflow.png

The built binary wheels and the source distribution are the "build artifacts" at the bottom.
You need to download both the wheels and the source distribution.
Save them on your computer, and unzip both files; you should have many wheel ``qutip-*.whl`` files, and two sdist files: ``qutip-*.tar.gz`` and ``qutip-*.zip``.
These are the same files that have just been uploaded to PyPI.


Monitoring progress (optional)
------------------------------

While the build is in progress, you can monitor its progress by clicking on its entry in the list below the "Run workflow" button.
You should see several subjobs, like the completed screen, except they might not yet be completed.

The "Verify PyPI deployment confirmation" should get ticked, no matter what.
If it fails, you have forgotten to choose the correct branch in the drop-down menu or you made a typo when confirming the correct branch, and you will need to restart this step.
You can check that the deployment instruction has been understood by clicking the "Verify PyPI deployment confirmation" job, and opening the "Compare confirmation to current reference" subjob.
You will see a message saying "Built wheels will be deployed" if you typed in the confirmation, or "Only building wheels" if you did not.
If you see "Only building wheels" but you meant to deploy the release to PyPI, you can cancel the workflow and re-run it after typing the confirmation.


.. _github:

Making a release on GitHub
++++++++++++++++++++++++++

This is all done through `the "Releases" section <https://github.com/qutip/qutip/releases>`_ of the ``qutip/qutip`` repository on GitHub.

- Click the "Draft a new release" button.
- Choose the correct branch for your release (e.g. ``qutip-4.5.X``) in the drop-down.
- For the tag name, use ``v<your-version>``, where the version matches the contents of the ``VERSION`` file.
  In other words, if you are releasing a micro version 4.5.3, use ``v4.5.3`` as the tag, or if you are releasing major version 5.0.0, use ``v5.0.0``.
- The title is "QuTiP <your-version", e.g. "QuTiP 4.6.0".
- For the description, write a short (~two-line for a micro release) summary of the reason for this release, and note down any particular user-facing changes that need special attention.
  Underneath, put the changelog you wrote when you did the documentation release.
  Note that there may be some syntax differences between the ``.rst`` file of the changelog and the Markdown of this description field.
- Drag-and-drop all the ``qutip-*.whl``, ``qutip-*.tar.gz`` and ``qutip-*.zip`` files you got after the build step into the assets box.
  You may need to unzip the files ``wheels.zip`` and ``sdist.zip`` to find them if you haven't already; **don't** upload those two zip files.

Click on the "Publish release" button to finalise.


.. _web:

Website
+++++++

This assumes that qutip.github.io has already been forked and familiarity with the website updating workflow.
The documentation need not be updated for every micro release.

Copying new files
-----------------

You only need to copy in new documentation to the website repository.
Do not copy the ``.whl``, ``.tar.gz`` or ``.zip`` files into the git repository, because we can access the public links from the GitHub release stage, and this keeps the website ``.git`` folder a reasonable size.

For all releases move (no new docs) or copy (for new docs) the ``qutip-doc-<MAJOR>.<MINOR>.pdf`` into the folder ``downloads/<MAJOR>.<MINOR>.<MICRO>``.

The legacy html documentation should be in a subfolder like ::

    docs/<MAJOR>.<MINOR>
    
For a major or minor release the previous version documentation should be moved into this folder. 

The latest version HTML documentation should be the folder ::

    docs/latest
    
For any release which new documentation is included
- copy the contents ``qutip-doc/_build/html`` into this folder. **Note that the underscores at start of the subfolder names will need to be removed, otherwise Jekyll will ignore the folders**. There is a script in the ``docs`` folder for this. 
https://github.com/qutip/qutip.github.io/blob/master/docs/remove_leading_underscores.py


HTML file updates
-----------------

- Edit ``download.html``

    * The 'Latest release' version and date should be updated.
    * The tar.gz and zip links need to have their micro release numbers updated in their filenames, labels and trackEvent javascript.
      These links should point to the "Source code" links that appeared when you made in the GitHub Releases section.
      They should look something like ``https://github.com/qutip/qutip/archive/refs/tags/v4.6.0.tar.gz``.
    * For a minor or major release links to the last micro release of the previous version will need to be moved (copied) to the 'Previous releases' section.

- Edit ``_includes/sidebar.html``

    * The 'Latest release' version should be updated. The gztar and zip file links will need the micro release number updating in the traceEvent and file name.
    * The link to the documentation folder and PDF file (if created) should be updated.

- Edit ``documentation.html``

    * The previous release tags should be moved (copied) to the 'Previous releases' section.

.. _cforge:

Conda-forge
+++++++++++

If not done previously then fork the qutip-feedstock:
https://github.com/conda-forge/qutip-feedstock

Checkout a new branch on your fork, e.g. ::

    $ git checkout -b version-4.0.2

Generate a new sha256 code from the gztar for this version, e.g. ::

    $ openssl sha256 qutip-4.0.2.tar.gz

Edit the ``recipe/meta.yaml`` file.
Change the version. Update the sha256 code. 
Check that the recipe package version requirements at least match those in the setup.cfg. 
Also ensure that the build number is reset ::

    build:
        number: 0

Push changes to your fork, e.g. ::

    $ git push --set-upstream origin version-4.0.2

Make a Pull Request.
This will trigger tests of the package build process.

If (when) the tests pass, the PR can be merged, which will trigger the upload of the packages to the conda-forge channel.
To test the packages, add the conda-forge channel with lowest priority ::

    $ conda config --append channels conda-forge

This should mean that the prerequistes come from the default channel, but the qutip packages are found in conda-forge.
