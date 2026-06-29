# Feedback

- The main ambiguity is the LaBraM link. The assignment points to
  `braindecode/Labram-Braindecode`, which is not packaged like the current
  Braindecode `from_pretrained(...)` models. It contains a raw
  `braindecode_labram_base.pt` checkpoint plus conversion/loading instructions.
  That artifact matches the older Braindecode 1.3.x LaBraM implementation, but
  it does not load cleanly against the newer Braindecode 1.4+/1.5+ LaBraM class:
  the state dict has missing/unexpected keys and positional-embedding shape
  mismatches.
- This creates a version conflict with CBraMod. The CBraMod model linked in the
  assignment is exposed through the newer Braindecode Hub path, and the CBraMod
  class is available in Braindecode 1.4+. So using old Braindecode for the linked
  LaBraM checkpoint conflicts with using new Braindecode for CBraMod.
- To keep the benchmark runnable with one modern dependency set, I used
  `braindecode/labram-pretrained` for LaBraM instead of the raw assignment-linked
  checkpoint. That repository is a maintained Braindecode Hub export with normal
  `from_pretrained(...)` files and loads directly in the current Braindecode
  package.
- For future versions of the assignment, it would help to state whether the
  exact linked LaBraM artifact is required or whether any maintained Braindecode
  LaBraM checkpoint is acceptable. If exact artifacts are required, providing a
  pinned Braindecode version or a small verified loading script for both models
  would remove this ambiguity.
