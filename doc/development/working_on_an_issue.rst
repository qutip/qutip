.. _development-working-on-an-issue:

***********************
How to work on an issue
***********************

An interesting issue can be a doorway into a codebase. You do not need to
understand the entire project to begin. This guide helps you take the next
steps: understand the question, explore the relevant code, choose an approach,
and share a change you can explain. AI tools can help you explore and understand
the code and the problem, and help you write code.

.. admonition:: Give yourself the chance to learn
   :class: tip

    Please resist the temptation to hand an issue to an AI agent and forward
    its patch straight to GitHub, or simply relay messages between the agent
    and reviewers. We have access to these tools too. What we value in your
    contribution is the understanding you develop and the judgment you bring
    to the design.

1. Find a place to start
========================

There are open issues you can work on. These are opportunities to get to know
the project by working on a real problem and helping shape its solution. The
goal is to learn how the project works, why its interfaces are designed as
they are, and how the code is structured.

Choose one that interests you. You can start with limited knowledge of the
project and build your understanding as you investigate. Some issues will
take time to explore, even when the eventual change is small.

Begin by reading the issue and its discussion. What is someone trying to
accomplish? Try the example if there is one, and ask questions when something
is unclear. You do not need a solution yet. You just need a starting point
for exploring together.

.. admonition:: The issue is a starting point for discovery
   :class: note

    Sometimes we open an issue because we have spotted a problem, before we
    have worked out the right solution. A suggested implementation, including
    one from a maintainer, is something you can investigate and improve.
    Finding a better approach is a welcome contribution.

2. Follow the example into the code
===================================

Start by finding the code behind the example, along with related tests and
documentation. If you use an AI tool, a useful first prompt is:

    "Help me understand how this example runs. Explain what each component is
    responsible for, and point me to related tests and documentation. Let's
    explore before making changes."

Open those files and follow one path from the public API to the implementation
and back to the result. If a function is confusing, try a small example to
see what it does. You can also ask an AI tool to walk through it with you,
then check its explanation against the code.

As the path becomes clearer, ask **why it was designed this way**. Nearby
code, Git history, and earlier discussions can reveal constraints that are
easy to miss when looking at one function. They can also show where an old
decision no longer fits the current needs.

A useful point to pause is when you can tell someone: "This component owns
the behavior, these callers depend on it, and this is where the issue appears."
You can keep investigating the parts you are still unsure about.

3. Where would the change belong?
=================================

Now consider how the solution would fit into what you have learned. Compare
possible approaches and look closely at the trade-offs. An AI tool can help
you explore these questions:

- **What can we reuse?** Look for existing functionality before adding a
  second place that does the same job.
- **Who should own the behavior?** A clear responsibility makes the change
  easier to follow and helps keep unrelated components independent.
- **Will this make future changes harder?** Consider whether a new option,
  special case, or fallback will make the code harder to understand, test,
  or extend over time.

Clear interfaces and focused modules make it easier to understand one part
without reconstructing the whole system. This helps both contributors and
AI tools, which also work with limited context.

Sometimes a seemingly small fix needs increasingly awkward workarounds.
For example, passing the same information through several unrelated
components may suggest that the responsibility belongs elsewhere. Pause and
consider whether a modest restructuring would help. Compare it with a direct
fix: what becomes clearer, what else must change, and could existing users
be affected? Keep the restructuring focused on the problem.

For substantial changes, discuss the direction in the issue before committing
to a full implementation. Share what you have learned and invite others to
fill in missing context. A small exploratory prototype can help answer a
design question and give that discussion concrete evidence.

4. Try the approach and check the behavior
==========================================

When trying an approach, choose a small example that will help you judge
whether it fits. Implement enough to try it out, then examine and revise the
result. If you use AI to help write code, make sure you understand what it
produces. Working code is evidence to consider, and the design still needs
your judgment.

For good first issues, QuTiP's policy asks you to write the implementation
yourself. You can use AI as a learning aid to understand the code.

Work in small steps so you can see what each change does. When an agent drafts
code, read the diff and trace your example through the changed path. If a
decision surprises you, ask why it is needed and explore alternatives. If the
implementation becomes awkward, revisit the design before adding more.

Read the tests with the same care. You can ask your tools to help, but check
what each test actually demonstrates. Would it catch the behavior going
wrong? Would it still pass if the implementation changed while preserving
that behavior? Passing tests are useful evidence when they protect the
expectations that matter.

When updating documentation, explain what users can do and what behavior
they should expect. Review AI-generated prose for accuracy and clarity, and
remove implementation details or development history that do not help users.

5. Help reviewers follow your reasoning
=======================================

A pull request (PR) invites others to examine your change with you. Explain
the problem, why you chose this approach, and how you checked it in your own
words. Give reviewers enough context to follow your reasoning even if they
have not read the issue discussion.

Every PR must include a completed **AI Tools Usage Disclosure**, as required
by `QuTiP's AI tools usage policy
<https://github.com/qutip/qutip/blob/master/CONTRIBUTING.md#ai-tools-usage-policy>`_.
Fill out this section of the `current PR template
<https://github.com/qutip/qutip/blob/master/.github/pull_request_template.md>`_
even if you did not use AI tools; in that case, explicitly state that none
were used. If you did use AI tools, name the tools or models and explain how
you used them.

Write your PR description and review replies yourself; translation tools are
welcome. Keep details that help someone assess the change, and cut repetition
and lengthy lists that obscure the reasoning. If a summary is too long for
you to read carefully, it needs another edit before you ask reviewers to
read it.

Review is another chance to learn. A question may uncover a constraint you
missed or help you explain a choice more clearly. You can use AI to investigate
a comment, then check the answer and respond from your own understanding.
