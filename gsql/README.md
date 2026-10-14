# GSQL

Every GSQL file of the graph `Mule_Pattern_Learner`. A file is named after the
responsibility its queries share, a query verb first after its own; the names are
constants of `contract.server`. [Queries](../docs/reference/queries.md) describes each
query; [Set up a graph](../docs/how-to/set-up-a-graph.md) covers a fresh graph and a
reused one.

| Folder | File | Defines | Installed by |
|---|---|---|---|
| `schema/` | `schema.gsql` | The graph and its types: fresh-graph DDL, run by a person, never on a populated graph | a person |
| | `loading.gsql` | The loading job `load_graph`, which loads PhantomLedger's export by column name | a person |
| | `scope_vertex.gsql` | The `Temporal_Training_Scope` vertex type and its membership edge | `mule install`, when the type is missing or outdated |
| `queries/` | `training_context.gsql` | `fetch_training_context`, the model's input | `mule install`, `mule train` |
| | `fourier64.gsql` | `encode_fourier64`, the subquery of the context query | the same |
| | `training_scope.gsql` | `create_training_scope`, `finalize_training_scope`, `list_scope_accounts` | the same |
| | `split_cutoffs.gsql` | `resolve_split_cutoffs` | the same |
| | `hub_accounts.gsql` | `list_hub_accounts` | the same |
| | `label_reveal.gsql` | `draw_reveal_uniforms`, `reveal_mule_labels` | the same |
| `evaluation/` | `ground_truth.gsql` | `read_ground_truth`, the oracle: for the audits and the diagnostics only | the same |

- **Installation** creates and installs only the queries whose text differs (comments,
  whitespace and case outside string literals aside), with their callers; `mule install`
  then drops the retired names still installed. A full install takes about 50 minutes,
  mostly the context query ([Installation](../docs/reference/queries.md#installation)).
- **The context query** is edited by hand like every other file. Its text names a
  contract (`CONTEXT_CONTRACT`), so a changed text needs a new one; the failing test
  prints it. Tests also keep its parameters and bounds equal to the Python contracts.
- **Writes**: only the creation and finalisation of the graph's one split (experiment
  membership) and the label reveal (label fields of the internal Accounts);
  relationships never change. Nothing deletes a split but its owner.
- **Labels**: only the reveal and `read_ground_truth` read the ground truth; no feature,
  cutoff or hub query reads a label attribute, and the population query reads only
  `pu_label` and its availability ([Labels](../docs/reference/labels.md)).
- **Schema changes** may invalidate compiled queries and positional loading jobs: verify
  and restore both. The installer applies only the scope vertex type, when it is missing
  or, by `mule install`, in place of an outdated one while no scope vertex uses it
  ([The scope types](../docs/reference/queries.md#the-scope-types)).
- **Checking**: `.gsqlroot` makes this folder the root `gsql-lsp` indexes, so a file
  checked alone, from the command line or an editor, is checked against `schema/`
  (`gsql-lsp check gsql/queries/training_context.gsql`).
