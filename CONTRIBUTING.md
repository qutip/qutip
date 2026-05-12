# Contributing to QuTiP Development

You are most welcome to contribute to QuTiP development by forking this repository and sending pull requests, or filing bug reports at the [issues page](https://github.com/qutip/qutip/issues).
You can also help out with users' questions, or discuss proposed changes in the [QuTiP discussion group](https://groups.google.com/g/qutip).
All code contributions are acknowledged in the [contributors](https://qutip.readthedocs.io/en/stable/contributors.html) section in the documentation.

For more information, including technical advice, please see the ["contributing to QuTiP development" section of the documentation](https://qutip.readthedocs.io/en/stable/development/contributing.html).


## AI Tools Usage Policy
We acknowledge the use of AI tools to improve efficiency and enhance quality of work. Contributors may use such tools, provided they adhere to the following guidelines:

### 1. Accountability

The human contributor is solely responsible for their contribution i.e. all the AI-generated outputs can be considered their own work. If you're submitting a Pull Request that includes AI-generated code or documentation:

- You are responsible for ensuring that code you submit meets the project's standards.
- You must fully understand every line of code in the submission.
- You must be able to explain the "why" behind the implementation during the review process.

### 2. Transparency

All Pull Requests must fill the **AI Tools Usage Disclosure** in the PR template. This disclosure is mandatory and must accurately reflect how AI Tools were used.

### 3. Copyright & Legal

By submitting a contribution to qutip, you agree to

1. Submit your contribution under the project's license.

2. The contribution does not violate any third-party rights or the terms of service of the AI provider. And does not include "regurgitated" code from libraries with incompatible licenses (e.g., GPL-licensed code) being suggested into our BSD-3-clause licensed project.

3. AI agents must not sign commits or be added to commit message trailer `Co-authored-by:` since copyright is fundamentally tied to the concept of human authorship as per the US Copyright law. You can instead use `Assisted-by: AI Model/Tool` as commit message trailer e.g. `Assisted-by: Cursor with Opus 4.6`.

### 4. Good Use Cases

- **Understand the codebase:** AI can be used to explore unfamiliar parts of the repository, summarize modules and clarify how components interact. It helps build the mental model of the project faster. 

- **Brainstorming and Design:**  AI may assist in generating ideas, evaluating different approaches and structuring designs. However, all final decisions regarding design and architecture should be independently validated and be made by the contributor.

- **Reviewing your implementation:**  AI can be used to spot potential bugs, suggest improvements and help clarify unexpected behavior in code. Contributors should verify the correctness and relevance of AI-generated feedback before applying it.

### 5. Prohibited Use

The following use cases are prohibited:

- **Communication:** In project communications (GitHub Issues, Discussion, PR descriptions and review comments), we personally expect to communicate directly with other humans not with automated systems. Use of translation tools is completely welcome.

- **Ban on Bots/Agents:** Fully autonomous or unsupervised AI agents (e.g. OpenClaw, SWE-agent) are not allowed to submit Pull Requests.

- **Stance on Good First Issues:** The main purpose of a “good first issue” is to allow new human contributors to get familiar with `qutip`, with the GitHub workflow and to engage with the community. Submitting a fully AI-generated PR defeats that purpose. AI can be used as a learning aid to understand the codebase, but the final implementation must be your own.

### 6. Enforcement

Maintainers reserve the right to close any Pull Request that violates this Policy. Maintainers may restrict participation or report users to GitHub for repeated violations of this policy.
