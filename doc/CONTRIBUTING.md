# Contributing to QuTiP Development

This assumes that you are working with Anaconda Python. This is not essential - instructions can be adapted. 

It also assumes some working knowledge of git and Github. This is essential for contributing to QuTiP. 
There are other resources for learning the necessary. The admin team will help.

Note that `~/` refers to your home directory.

## Create conda environment

You will want to create a conda environment (env) for qutip development. 
Full instuctions for this are in the qutip docs
http://qutip.org/docs/latest/installation.html

This would be typical:

```
conda create -n qutip-dev-py3 python=3 numpy scipy cython matplotlib pytest pytest-cov jupyter notebook spyder
```

If you wanting to build the documentation, then you will need more libraries, see:
https://github.com/qutip/qutip-doc/blob/master/README.md

## Forking and cloning repositories
Most likely you will be wanting to contribute to the core code of qutip. 
There are a few other repositories (repos) that support the project. 
Hence you may wish to create a parent directory called something like `qutip-project`.
Each time you clone a repo into it you will create a sub-directory, 
so you should end up with a directory structure that looks a bit like this:

```
~/qutip-project/qutip
               /qutip-notebooks
               /qutip-docs
               /qutip-github-io
```
You will need to create your own fork of the repo(s) that you wish to contribute to. 
You do this from the main page of the repo. They are all accessible from:
https://github.com/qutip

Your fork is a copy of the main repo, on which you can make edits, 
which you can then propose as changes by making a pull request.

### qutip core
Once you have made a fork of the main qutip repo, then you will have a repo like:
https://github.com/githubusername/qutip

To make a local copy that you can edit run this in a terminal whilst in your `qutip-project` directory:

```
git clone https://github.com/githubusername/qutip
```

You will need to install qutip in develop mode. First activate your conda env, e.g.

```
source activate qutip-dev-py3
```

Then move to the qutip directory and install:
```
cd qutip
python setup.py develop
```

### other qutip repos
You can fork and clone the other qutip repos in the same way. There is no need to run any install for them.

## Adding remotes
You will need to add the parent repositories as 'remotes'. The standard is to use the name 'upstream'.
You need to be inside the repos directory. As an example, for the qutip core:

```
cd ~/qutip-project/qutip
git remote add upstream https://github.com/qutip/qutip
```

You may also want to add remotes for other user's forks if you want to checkout their branches.

## git branches
A git branch enables you to work on something without effecting the master files until you want to.
Should create a branch on your repo whenever you want to try something new. 
You can have many different branches, but you work on one at a time.

### pulling the upstream master
Unless you have very recently pulled the upstream master into your own, 
then you will want to do this before creating a new branch from it.

```
git checkout master
git pull upstream master
git push
```

This brings any changes from the main repo into your fork, and then updates your fork.

### creating a development branch
To create a branch for your new development:

```
git checkout -b my-branch_name
```

The branch name, in this example 'my-branch_name', should be something descriptive, 
that would identifiable with the feature in the git log.

### committing changes
Whenever you feel that you have reached some milestone (or just the end of the day), 
then commit your changes.

```
git commit -a
```
You will need to add a commit message that describes what you have done since the last commit.

### pushing branches
Committing is only a local operation. When you want to share your branch, or just back it up, 
you should 'push' it to the repo.

```
git push
```
This will not be enough for the first push of a branch, but git will tell you what to add.

### checking out remote branches
If you want to checkout a copy of someone else's branch, then (assuming you have added the remote) first:
```
git fetch remote_name
```
Then:
```
git checkout -b branch_name remote_name/branch_name
```
This assumes you know the branch name. You can see all fetched branch names with:

```
git branch -v
```

### switching branches
You can see all your local branches using:
```
git branch
```

You can switch between branches using:
```
git checkout branch_name
```

You cannot switch to another branch until you have committed any changes on the current branch.

## Making a pull request
If you have recently made a push to your repo, then you will see an option to make a pull request (PR). 
You make the pull request when you believe that your contribution is ready to be considered by the admin team.
You can continue to add and push commits onto the branch after you have made the PR.
The PR should have a descriptive title. A description should be given that explains what it is for, 
and hence why it should be merged into the master code.

## Coding standards
Contributions to qutip should follow the pep8 style:
https://www.python.org/dev/peps/pep-0008/

Docstrings are processed by numpydoc, and hence should be formatted in line with:
http://numpydoc.readthedocs.io/en/latest/format.html

You should attempt to use parameter and attribute names that are used elsewhere in the library where possible.

Unit tests should be included for any new core code.

## Example notebooks
Any new features should be supported by a example [Jupyter](http://jupyter.org/) notebook.
This should be submitted as a PR on the qutip-notebooks repo and linked from the qutip PR.

## The review process
The qutip admin team will review any new PRs. Feedback will be provided, changes may be required. 
If it is appropriate, and when it is deemed ready, it will be merged into the master branch.

