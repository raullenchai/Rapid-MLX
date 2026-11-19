# Research Loop — product context

A reusable local autoresearch workbench, organized by project. A researcher
opens Projects, chooses a template and goal, discusses approaches with a local
model, adds testable hypotheses, then starts a fixed-budget experiment loop. Results,
code, logs, and discussion stay together on the filesystem.

The project has three stages: Discuss → Experiments → Results. The smaller
five-step research strip places it within a wider research process; question and
writing remain in the researcher's tools. Literature is attached per project in
Discuss and grounds discussion and experiment planning. Research Loop does not
write papers. It supports multiple named local model endpoints.

This is an Operate surface for researchers with unpublished work. Preserve the
scientific figure sheet: Public Sans, cool mineral surfaces, broad white plotting
space, and semantic verdict colours. The main experiment chart plots metric
against experiment number with points coloured by hypothesis, labelled improvements, and a
running-best step line. The selected experiment’s curve and the discussion support it.
Results expose every recorded attempt and its reproducible artifacts.

All measurements and learned summaries come from events, with task metadata
supplying the metric name, unit, direction, noise threshold, and goal. No invented
results, generated scientific conclusions, or unsupported confidence claims.
Grokking and pure-Python sort-speed exercise higher- and lower-is-better projects;
web code must not assume a particular research domain.

Primary use is 1920×1080, including a short grant demonstration video; 1440×900
must also work. Experiments occupies one desktop viewport with internal scrolling.
Keyboard focus, meaningful empty states, safe markdown rendering, failure recovery,
and model/queue visibility are part of the working product.

Discussion produces many hypotheses. A hypothesis is a testable claim about the
system with why and prediction; each experiment is one concrete change that tests
it across all seeds under the fixed budget. Experiments report improved / worse /
no effect / crashed. Hypotheses aggregate that evidence into untested / testing /
supported / refuted / inconclusive; the model cannot assert a conclusion. Any
improvement supports a claim immediately; after the task’s experiment limit
(default 2), no improvement refutes it, or leaves it inconclusive if all crashed.
Researcher-added untested hypotheses go first. Concluded hypotheses receive no
more experiments. Discuss keeps its goal/discussion/right-rail shape; the rail
lists hypotheses and accepts new claims. Experiments groups attempts under
hypotheses; Results leads with an expandable hypothesis ledger and groups what
we learned by claim. Projects reports N hypotheses · M experiments. Old runs
remain readable without migration or writes to their artifacts.
